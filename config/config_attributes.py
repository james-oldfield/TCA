directions = {
    'pggan_celebahq1024': {
        'linear_attributes': {
            'male': [(1, -75), (0, 0), (0, 0)],
            'blonde': [(3, 75), (0, 0), (0, 0)],
            'young': [(4, 75), (0, 0), (0, 0)],
            'pale': [(11, 100), (0, 0), (0, 0)],
            'pitch': [(0, 0), (3, -30), (0, 0)],
            'yaw': [(0, 0), (0, 0), (1, -15)],
        },
        'multilinear_attributes': {
            'thick': (2, -600),
            'forehead': (6, 600),
            'skew': (5, 600),
        }
    },

    'pggan_car256': {
        'linear_attributes': {
            'blue': [(1, 50), (0, 0), (0, 0)],
            'red': [(9, 50), (0, 0), (0, 0)],
            'pitch': [(0, 0), (3, -15), (0, 0)],
            'yaw': [(0, 0), (0, 0), (1, -5)],
        },
    },

    'stylegan_car512': {
        'linear_attributes': {
            'red': [(0, 20), (0, 0), (0, 0)],
            'white_truck': [(2, 50), (0, 0), (0, 0)],
            'rotation': [(0, 0), (1, -10), (0, 0)],
            'size': [(0, 0), (0, 0), (2, 40)],
        },
        'multilinear_attributes': {
            'stretch_rear': (1, 1000),
        }
    },

    'stylegan_animeface512': {
        'linear_attributes': {
            'illumination': [(0, 20), (0, 0), (0, 0)],
            'dark': [(1, 20), (0, 0), (0, 0)],
            'pink': [(2, 20), (0, 0), (0, 0)],
            'blonde': [(5, 20), (0, 0), (0, 0)],
            'pose': [(0, 0), (0, 0), (0, 5)],
        },
    },

    'stylegan_ffhq1024': {
        'multilinear_attributes': {
            'thick': (2, 300),
            'long': (5, 300),
        }
    },

}