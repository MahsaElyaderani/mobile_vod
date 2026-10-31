VID_CLASSES = (
    ('airplane', 'n02691156'), ('antelope', 'n02419796'),
    ('bear', 'n02131653'), ('bicycle', 'n02834778'), ('bird', 'n01503061'),
    ('bus', 'n02924116'), ('car', 'n02958343'), ('cattle', 'n02402425'),
    ('dog', 'n02084071'), ('domestic_cat', 'n02121808'),
    ('elephant', 'n02503517'), ('fox', 'n02118333'),
    ('giant_panda', 'n02510455'), ('hamster', 'n02342885'),
    ('horse', 'n02374451'), ('lion', 'n02129165'), ('lizard', 'n01674464'),
    ('monkey', 'n02484322'), ('motorcycle', 'n03790512'),
    ('rabbit', 'n02324045'), ('red_panda', 'n02509815'),
    ('sheep', 'n02411705'), ('snake', 'n01726692'), ('squirrel', 'n02355227'),
    ('tiger', 'n02129604'), ('train', 'n04468005'), ('turtle', 'n01662784'),
    ('watercraft', 'n04530566'), ('whale', 'n02062744'), ('zebra', 'n02391049'),
)

def vid_to_dataset_labels(dataset):
    if dataset.num_classes != 30:
        raise ValueError('The released fork checkpoints require all 30 VID category definitions.')
    aliases = {}
    for label, (name, synset) in enumerate(VID_CLASSES, 1):
        aliases[name] = label
        aliases[synset] = label
    mapping = {}
    for dataset_label, name in dataset.label_to_name.items():
        canonical = name.strip().lower().replace(' ', '_')
        if canonical not in aliases:
            raise ValueError(
                f'Unknown VID category name {name!r}; use the canonical class name '
                'or its VID synset ID. Category IDs alone do not define class order.'
            )
        fork_label = aliases[canonical]
        if fork_label in mapping:
            raise ValueError(f'Duplicate category for fork label {fork_label}.')
        mapping[fork_label] = dataset_label
    if set(mapping) != set(range(1, 31)):
        raise ValueError('The categories must map one-to-one to all 30 fork classes.')
    return mapping
