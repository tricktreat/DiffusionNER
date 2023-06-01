import copy
import multiprocessing as mp
import os
import re
import time
import random
import json
import pynvml
import numpy as np

pynvml.nvmlInit()

def process_configs(target, arg_parser):
    args, _ = arg_parser.parse_known_args()
    ctx = mp.get_context('fork')

    subprocess=[]
    if "ALL_GPU" in os.environ:
        all_gpu_queue = list(map(int, os.environ["ALL_GPU"].split(",")))
    else:
        all_gpu_queue = [0, 1, 2, 3, 4, 5, 6, 7]
    gpu_queue = []
    waittime = 420
    gpu_just_used = []
    for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
        if "eval" in run_args.label:
            waittime = 90
            if "genia" in run_args.dataset_path:
                waittime = 180
            if "fewnerd" in run_args.dataset_path:
                waittime = 240
            if "ontonotes" in run_args.dataset_path:
                waittime = 240
            if "msra" in run_args.dataset_path:
                waittime = 240
        if run_args.seed==-1:
            run_args.seed=random.randint(0,1000)
        # debug
        if run_args.debug:
            target(run_args)
        while not run_args.cpu and (len(gpu_queue)==0 or len(gpu_queue)<run_args.world_size):
            gpu_queue = []
            candidate_gpu = list(set(all_gpu_queue) - set(gpu_just_used))
            for index in candidate_gpu:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    free = meminfo.free//(1024*1024)
                    if run_args.world_size>0:
                        gpu_queue.extend([(index, free)])
                    elif (("eval" in run_args.label)):
                        required_mem = 11000
                    else:
                        required_mem = 12000
                    cur_free = free
                    for i in range(free//required_mem):
                        gpu_queue.append((index, cur_free))
                        cur_free = cur_free - required_mem
                except Exception as e:
                    pass
            gpu_queue = sorted(gpu_queue, key=lambda x:x[1], reverse=True)
            print(dict(set(gpu_queue)))
            if len(gpu_queue)<run_args.world_size:
                print(f"Need {run_args.world_size} GPUs for DDP Training, but only {len(gpu_queue)} free devices: {gpu_queue}. Waiting for Free GPU ......")
                time.sleep(waittime)
                gpu_just_used = []
            elif len(gpu_queue)==0:
                print("Need 1 GPU for Normal Training, All are busy. Waiting for Free GPU ......")
                time.sleep(waittime)
                gpu_just_used = []
            else:
                print("Avaliable devices: ",list(map(lambda x:x[0],gpu_queue)))
            # gpu_just_used = []
        # CPU Training:
        if run_args.cpu:
                print("########### Using CPU Training ###########")
                print("Using Random Seed", run_args.seed)
                p = ctx.Process(target=target, args=(run_args,))
                subprocess.append(p)
                p.start()
                time.sleep(1)
        # GPU Training
        else:
            # GPU DDP Training
            if run_args.world_size != -1:
                print("########### Using GPU DDP Training ###########")
                # print("Using devices: ", gpu_queue)
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str,list(set(map(lambda x:x[0],gpu_queue)))))
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
                
                for local_rank in range(run_args.world_size):
                    gpu_just_used.append(gpu_queue[0][0])
                    gpu_queue = gpu_queue[1:]
                    run_args.local_rank = local_rank
                    print("Using Random Seed", run_args.seed)
                    p = ctx.Process(target=target, args=(run_args,))
                    subprocess.append(p)
                    p.start()
                time.sleep(1)
            # GPU Normal Training
            if run_args.world_size == -1:
                print("########### Using GPU Normal Training ###########")
                device_id = gpu_queue[0][0]
                run_args.device_id = device_id
                gpu_just_used.append(run_args.device_id)
                gpu_queue.remove(gpu_queue[0])
                print("Using devices: ", run_args.device_id)
                print("Using Random Seed", run_args.seed)
                p = ctx.Process(target=target, args=(run_args,))
                subprocess.append(p)
                p.start()
                time.sleep(1)

    list(map(lambda x:x.join(),subprocess))

def _read_config(path):
    lines = open(path).readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    config_list = []
    for k, v in config.items():
        if k == "config":
            continue
        if v == "None":
            continue
        if v.startswith("["):
            v = v[1:-1].replace(",", "")
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + [v])
    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    _print = (lambda x: print(x)) if verbose else lambda x: x

    if args.config:
        config = _read_config(args.config)

        for run_repeat, run_config in config:
            print("-" * 50)
            print("Config:")
            # print(run_config)

            args_copy = copy.deepcopy(args)
            run_config=copy.deepcopy(run_config)
            config_list = _convert_config(run_config)
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)

            run_args_list = []
            # batch eval
            if run_args.label=="batch_eval_flag":
                save_path=run_args.model_path
                # save_model_type = run_args.save_model_type
                for dirpath,dirnames,filenames in sorted(os.walk(save_path),key=lambda x:x[0]):
                    if "final_model" in dirpath:
                        print(f"Find model checkpoint @ {dirpath}")
                        dataset_name=re.match(".*/(.*?)_train",dirpath).group(1)
                        args_path="/".join(dirpath.split("/")[:-1])+"/args.json"
                        args_dict=json.load(open(args_path))

                        for k, v in args_dict.items():
                            if k not in ["seed", "boundary_threshold", "cls_threshold", "entity_threshold", "sampling_timesteps", "num_proposals", "span_renewal", "step_ensemble", 
                            "eval_batch_size", "no_overlapping", "no_partial_overlapping", "no_duplicate", "store_predictions", "store_examples", "sample_dist_type"]:
                                setattr(run_args, k, v)

                        run_args.label= dataset_name+"_eval"
                        if "train_dev" in args_dict["train_path"]:
                            run_args.dataset_path =  args_dict["train_path"].replace("train_dev","test")
                        else:
                            run_args.dataset_path =  args_dict["train_path"].replace("train","test")
                        run_args.model_path=dirpath
                        run_args.tokenizer_path=dirpath
                        run_args.log_path = "/".join(dirpath.split("/")[:-3])
                        run_args.save_code = False


                        if run_args.num_proposals == -1:
                            for num_proposals in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]:
                                run_args_instance = copy.deepcopy(run_args)
                                if run_args_instance.sampling_timesteps == -1:
                                    run_args_instance.sampling_timesteps = 5
                                run_args_instance.num_proposals = num_proposals
                                run_args_instance.label = run_args_instance.label + "_num_proposals_" + dirpath.split("/")[-2]
                                run_args_list.append(run_args_instance)

                        
                        if run_args.sampling_timesteps == -1:
                            for sampling_timesteps in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                                run_args_instance = copy.deepcopy(run_args)
                                if run_args_instance.num_proposals == -1:
                                    run_args_instance.num_proposals = 60
                                run_args_instance.sampling_timesteps = sampling_timesteps
                                run_args_instance.label = run_args_instance.label + "_sampling_timesteps_" + dirpath.split("/")[-2]
                                run_args_list.append(run_args_instance)

                        if run_args.entity_threshold == -1:
                            for entity_threshold in np.arange(2.3, 2.9, 0.05):
                                run_args_instance = copy.deepcopy(run_args)
                                run_args_instance.entity_threshold = entity_threshold
                                run_args_instance.label = run_args_instance.label + "_entity_threshold_" + dirpath.split("/")[-2]
                                run_args_list.append(run_args_instance)

                        if run_args.cls_threshold == -1 and run_args.boundary_threshold != -1:
                            for cls_threshold in np.arange(0, 1, 0.1):
                                run_args_instance = copy.deepcopy(run_args)
                                run_args_instance.cls_threshold = cls_threshold
                                run_args_instance.label = run_args_instance.label + "_cls_threshold_" + dirpath.split("/")[-2]
                                run_args_list.append(run_args_instance)

                        if run_args.boundary_threshold == -1 and run_args.cls_threshold != -1:
                            for boundary_threshold in np.arange(0, 1, 0.1):
                                run_args_instance = copy.deepcopy(run_args)
                                run_args_instance.boundary_threshold = boundary_threshold
                                run_args_instance.label = run_args_instance.label + "_boundary_threshold_" + dirpath.split("/")[-2]
                                run_args_list.append(run_args_instance)
                                
                        if run_args.cls_threshold == -1 and run_args.boundary_threshold == -1:
                            for cls_threshold in np.arange(0, 1, 0.1):
                                for boundary_threshold in np.arange(0, 1, 0.1):
                                    run_args_instance = copy.deepcopy(run_args)
                                    run_args_instance.cls_threshold = cls_threshold
                                    run_args_instance.boundary_threshold = boundary_threshold
                                    run_args_instance.label = run_args_instance.label + "_cls_boundary_threshold_" + dirpath.split("/")[-2]
                                    run_args_list.append(run_args_instance)

                        if run_args.cls_threshold != -1 and run_args.boundary_threshold != -1 and run_args.entity_threshold != -1 and run_args.sampling_timesteps != -1 and run_args.num_proposals != -1:
                            run_args_list.append(copy.deepcopy(run_args))

            else:
                run_args_list.append(run_args)

            for run_args in run_args_list:
                print(run_args)
                print("Repeat %s times" % run_repeat)
                print("-" * 50)
                candidate_seed = [488, 704, 763, 802, 895]
                
                for iteration in range(run_repeat):
                    _print("Iteration %s" % iteration)
                    _print("-" * 50)
                    
                    # seed = random.choice(candidate_seed)
                    if run_args.seed == -1:
                        seed = candidate_seed[0]
                        candidate_seed.remove(seed)
                        run_args.seed = seed

                    yield copy.deepcopy(run_args), run_config, run_repeat

    else:
        yield args, None, None
