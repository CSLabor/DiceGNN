min_makespan = 10000000
record_plan = None
additional_trials = 0

def simulate(cpu_buffer_size, gpu_buffer_size, t_cpu, t_dma, t_uva, t_model, t_cache):
    """
    simulate the executing of one "overlap"
    the training of one epoch consists of several "overlaps"
            <------------------ overlap span ---------------->
    GPU:    uva uva ...  uva pop+model ... sync+pop+model ... 
                             <------- buffers flushing ------>
            <-uva buf fill->
    PCI:    uva uva ... uva dma dma ... ... ... ... ... dma
            <-cpu buf fill->
    CPU:    cpu cpu ... ... cpu cpu ... ... ... ... ... cpu
                            <-cpu buf fill for next overlap->

    return: (feedback, sched_plan, converge)
    * feedback: 1 for too much CPU workload and 0 for too much GPU workload
    * converge: bool
    """
    assert min(cpu_buffer_size, gpu_buffer_size) > 0

    cpu_min_time = cpu_buffer_size * t_cpu
    pcie_min_time = cpu_buffer_size * t_dma + gpu_buffer_size * t_uva
    gpu_min_time = gpu_buffer_size * (t_uva + t_cache) + (gpu_buffer_size + cpu_buffer_size) * t_model
    max_makespan = 2*int(cpu_min_time + pcie_min_time + gpu_min_time)

    # (type, id, start_ts, end_ts)
    cpu_activities = []
    pcie_activities = []
    gpu_activities = []

    for i in range(cpu_buffer_size):
        cpu_activities.append(("cpu", i, i*t_cpu, (i+1)*t_cpu))
    for i in range(gpu_buffer_size):
        pcie_activities.append(('uva', i, i*t_uva, (i+1)*t_uva))
        gpu_activities.append(('uva', i, i*t_uva, (i+1)*t_uva))

    # simulate the overlap
    for ts in range(0, max_makespan):
        last_cpu_act = cpu_activities[-1]
        last_pcie_act = pcie_activities[-1]
        last_gpu_act = gpu_activities[-1]

        if ts < min(last_cpu_act[-1], last_pcie_act[-1], last_gpu_act[-1]):
            continue

        if (last_gpu_act[0] == "model_c") and (last_gpu_act[1] == cpu_buffer_size-1) and (ts > last_gpu_act[-1]):
            # finish
            break

        # check pcie and assign dma workload
        if ts > last_pcie_act[-1]:
            # should begin dma transfer of one prepared CPU batch
            if last_pcie_act[0] == 'uva':
                cur_dma_id = 0
            else:
                assert "dma" == last_pcie_act[0]
                cur_dma_id = last_pcie_act[1] + 1

            if cur_dma_id <= cpu_buffer_size - 1:
                # need to ensure that this cpu batch has finished, otherwise do nothing and wait
                if ts > cpu_activities[cur_dma_id][-1]:
                    start_ts = max(cpu_activities[cur_dma_id][-1], last_pcie_act[-1])
                    pcie_activities.append(("dma", cur_dma_id, start_ts, start_ts+t_dma))

        # check gpu and assign model workload
        if ts > last_gpu_act[-1]:
            if last_gpu_act[0] == 'uva':
                gpu_activities.append(("model_g", 0, last_gpu_act[-1], last_gpu_act[-1]+t_model+t_cache))
                continue

            if last_gpu_act[0] == "model_g":
                cur_model_id = last_gpu_act[1] + 1
                cur_start_ts = last_gpu_act[-1]
                if cur_model_id <= gpu_buffer_size - 1:
                    gpu_activities.append(("model_g", cur_model_id, cur_start_ts, cur_start_ts+t_model+t_cache))
                    continue

            assert cur_model_id == gpu_buffer_size or last_gpu_act[0] == "model_c"
            if last_gpu_act[0] == "model_c":
                cur_model_id = last_gpu_act[1] + 1
            else:
                cur_model_id = 0

            if cur_model_id <= cpu_buffer_size - 1:
                # need to ensure the dma transfer of this batch has finished
                if len(pcie_activities) >= gpu_buffer_size+cur_model_id+1 and ts > pcie_activities[cur_model_id+gpu_buffer_size][-1]:
                    start_ts = max(pcie_activities[cur_model_id+gpu_buffer_size][-1], last_gpu_act[-1])
                    gpu_activities.append(("model_c", cur_model_id, start_ts, start_ts+t_model))

    # examine the current overlap scheduling
    #print(cpu_activities)
    #print(pcie_activities)
    #print(gpu_activities)

    global min_makespan, record_plan, additional_trials
    cur_makespan = gpu_activities[-1][-1]
    if cur_makespan < min_makespan:
        min_makespan = cur_makespan
        record_plan = (cpu_buffer_size, gpu_buffer_size)
        additional_trials = 0
        #print("global min!", min_makespan, record_plan)
    else:
        additional_trials += 1

    if additional_trials > 20:
        return None, record_plan, True


    gpu_util = sum([end - start for start, end in [x[-2:] for x in gpu_activities]]) / gpu_activities[-1][-1]
    cpu_util = sum([end - start for start, end in [x[-2:] for x in cpu_activities]]) / gpu_activities[-1][-1]
    if cpu_util > gpu_util:
        return 1, (cpu_buffer_size, gpu_buffer_size), False
    return 0, (cpu_buffer_size, gpu_buffer_size), False


if __name__ == "__main__":
    print(simulate(10,10,15,10,12,5,0))
