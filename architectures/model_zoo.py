# python3.7
"""Model zoo."""

# pylint: disable=line-too-long

MODEL_ZOO = {
    # PGGAN official.
    'pggan_celebahq1024': dict(
        gan_type='pggan',
        resolution=1024,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EW_3jQ6E7xlKvCSHYrbmkQQBAB8tgIv5W5evdT6-GuXiWw?e=gRifVa&download=1',
    ),
    'pggan_car256': dict(
        gan_type='pggan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EfGc2we47aFDtAY1548pRvsByIju-uXRbkZEFpJotuPKZw?e=DQqVj8&download=1',
    ),

    # StyleGAN official.
    'stylegan_ffhq1024': dict(
        gan_type='stylegan',
        resolution=1024,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EdfMxgb0hU9BoXwiR3dqYDEBowCSEF1IcsW3n4kwfoZ9OQ?e=VwIV58&download=1',
    ),
    'stylegan_bedroom256': dict(
        gan_type='stylegan',
        resolution=256,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Ea6RBPddjcRNoFMXm8AyEBcBUHdlRNtjtclNKFe89amjBw?e=Og8Vff&download=1',
    ),
    'stylegan_car512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EcRJNNzzUzJGjI2X53S9HjkBhXkKT5JRd6Q3IIhCY1AyRw?e=FvMRNj&download=1',
    ),

    # StyleGAN third-party.
    'stylegan_animeface512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EWDWflY6lBpGgX0CGQpd2Z4B5wTEVamTOA9JRYne7zdCvA?e=tOzgYA&download=1',
    ),
    'stylegan_artface512': dict(
        gan_type='stylegan',
        resolution=512,
        url='https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/Eca0OiGqhyZMmoPbKahSBWQBWvcAH4q2CE3zdZJflp2jkQ?e=h4rWAm&download=1',
    ),
}

# pylint: enable=line-too-long
