from dataset import DDFF12Loader_Val, Uniformat, InfinigenDefocus, Zedd, HAMMER

f_inf = float("inf")
f_12 = 12.0
f_5_6 = 5.6
f_4_0 = 4.0
f_2_8 = 2.8
f_2_0 = 2.0
f_1_4 = 1.4
f_1_0 = 1.0
f_0_5 = 0.5

val_loader_configs = {
    'ibims_F1_4_adaptive_fd': {
        'dataloader': Uniformat,
        'dataset_name': 'iBims',
        'fnumber': f_1_4,
        'depth_dependent_fd_list': True,
        'eval_in_disparity_space': False,
        'psf_type': 'gauss',
        'focal_stack_size': 5,
    },
    'infinigen_defocus_F1_4_fixed_fd_0_8,1_7,3_0,4_7,8_0': {
        'dataloader': InfinigenDefocus,
        'dataset_name': 'InfinigenDefocus',
        'fnumber': f_1_4,
        'fd_list': [0.8, 1.7, 3.0, 4.7, 8.0],
        'eval_in_disparity_space': False,
        'use_focus_stack_from_dataset': True,
    },
    'diode_F1_4_adaptive_fd': {
        'dataloader': Uniformat,
        'dataset_name': 'DIODE',
        'fnumber': f_1_4,
        'depth_dependent_fd_list': True,
        'eval_in_disparity_space': False,
        'psf_type': 'gauss',
        'focal_stack_size': 5,
    },
    'zedd_F2_8_fixed_fd_0_2_4_6_8': {
        'dataloader': Zedd,
        'dataset_name': 'Zedd',
        'fnumber': f_2_8,
        'fd_list': [0, 2, 4, 6, 8], # This datases uses indices instead of actual fd values
        'eval_in_disparity_space': False,
        'use_focus_stack_from_dataset': True,
    },
    'zedd_test_F2_8_fixed_fd_0_2_4_6_8': {
        'dataloader': Zedd,
        'dataset_name': 'Zedd',
        'fnumber': f_2_8,
        'fd_list': [0, 2, 4, 6, 8], # This datases uses indices instead of actual fd values
        'eval_in_disparity_space': False,
        'use_focus_stack_from_dataset': True,
    },
    'ddff12_val': {
        'dataloader': DDFF12Loader_Val,
        'dataset_name': 'DDFF12Loader_Val',
        'fnumber': None, # no fnumber needed since we use fd list/focus stack from dataset
        'fd_list': None,  # use fd list from dataset
        'eval_in_disparity_space': True,
        'use_focus_stack_from_dataset': True,
    },
    'hammer_F1_4_adaptive_fd': {
        'dataloader': HAMMER,
        'dataset_name': 'HAMMER',
        'fnumber': f_1_4,
        'depth_dependent_fd_list': True,
        'eval_in_disparity_space': False,
        'psf_type': 'gauss',
        'focal_stack_size': 5,
    },

}
