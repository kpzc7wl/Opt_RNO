# Opt_RNO


### Dataset
https://drive.google.com/drive/folders/1ASQPuYoVd-yLQxVISs7NOdMrUgGCX635?usp=drive_link

### Training


To train RNO, parameters could be updated using argparser or modifying args.py file

```python
python train_geo.py --gpu 0 --dataset micro2d_sens_30x40 --use-normalizer unit  --normalize_x unit --component all --sobolev \
--loss-name rel2 --epochs 100 --batch-size 8 --model-name RNO \
--optimizer AdamW --weight-decay 0.00005   --lr 0.001 --lr-method cycle  \
--grad-clip 1000.0 --n-hidden 128 --n-layers 3  --use-tb 1 \
--seed 2024 --attn-type virtualfourier --n-head 8 --modes 16 --sample-rate 1 --gamma 1 --noref 0.3 --lamb 0.9 --comment _vf_ST0.9
```

Use ``--sobolev`` to enable Sobolev training with sensitivity data.  

Use``--attn-type virtualfourier`` to enable virtualfourier layer.  



### Test Samples
![Microreactor](/figure/micro_gt.svg)
![Microreactor_vf](/figure/micro_vf.svg)
![Microreactor_vf_noref](/figure/micro_vf_noref.svg)

![Fuelcell](/figure/fuelcell_gt.svg)
![Fuelcell_vf](/figure/fuelcell_vf.svg)
![Fuelcell_vf_noref](/figure/fuelcell_vf_noref.svg)

![Inductor](/figure/inductor_gt.svg)
![Inductor_vf](/figure/inductor_vf.svg)
![Inductor_vf_noref](/figure/inductor_vf_noref.svg)