# DEN

The official implementation of the paper '[Dynamic Information Enhancement for Video Classification](https://www.sciencedirect.com/science/article/pii/S0262885621001499?casa_token=BkD6H72_JVQAAAAA:asnY07Bf2mN4_xqrHSYDT9qhtCynWfIyp1iMBFjpfU2GRLpFja-OPcTS4GVtPMCu_x7Dc44eVQ)'.



## Data preparation

We  recommend to [TSM](https://github.com/mit-han-lab/temporal-shift-module) for dataset preparation.



## Training & Testing

using this command to train:

```
python main.py something RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.005 --lr_steps 30 45  --epochs 50 --batch-size 32 -j 8 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=4 --shift_place=blockres --comu_type motion_replace_A --npb --add_se
```



using this command to test:

```
python test_models.py something --weights=your_checkpoint --test_segments=8 --test_crops=1 --batch_size=96 --comu_type motion_replace_A --add_se  -j 8 
```



if adopt **2** clips × **3** crops with full resoluion (256×256) to ensemble performance, using this commands:

```
python test_models.py something --weights=your_checkpoint --test_segments=8 --test_crops=3 --batch_size=96 --comu_type motion_replace_A --add_se  -j 8 --twice_sample --full_res
```
