from train_unet_demo import build_args,cli_main

if __name__ == '__main__':
    challenges = ['SingleCoil','MultiCoil']
    tasks = ['Cine']
    sub_tasks = ['all']
    accelerations = [4,8,10]
    chans = [32,64,128,256]
    challenges = ['MultiCoil']
    chans = [128]
    for challenge in challenges:
        for task in tasks:
            for sub_task in sub_tasks:
                for acceleration in accelerations:
                    for chan in chans:
                        args = build_args(challenge = challenge,
                                   task = task,
                                   sub_task = sub_task,
                                   acceleration = acceleration,
                                   chan = chan)
                        cli_main(args)