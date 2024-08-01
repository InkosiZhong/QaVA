# QaVA: Query-aware Video Analysis Framework Based on Data Access Pattern

# Installation
1.Install [SWAG](https://github.com/stanford-futuredata/swag-python), [BlazeIt](https://github.com/InkosiZhong/blazeit) (a modified version) and [SUPG](https://github.com/stanford-futuredata/supg).
```
git clone https://github.com/stanford-futuredata/swag-python.git
cd swag-python/
conda install -c conda-forge opencv
pip install -e .
cd ..

git clone https://github.com/stanford-futuredata/blazeit.git
cd blazeit/
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge pyclipper
pip install -e .
cd ..

git clone https://github.com/stanford-futuredata/supg.git
cd supg/
pip install pandas feather-format
pip install -e .
cd ..
```
2.Install tiny-cuda-nn
```
cd thirdparty/tiny-cuda-nn-1d/bindings/torch
python setup.py install
```
3.Install QaVA
```
git clone https://github.com/inkosizhong/QaVA.git
cd QaVA
pip install -r requirements.txt
pip install -e .
```

# Usage

We provide code for `amsterdam` video dataset. You can download the `amsterdam` video dataset [here](https://drive.google.com/drive/folders/1mX3Z7ydI-mqTjxqchekDqQNUBRvmJyVV?usp=share_link). Download the `2017-04-10.zip` and `2017-04-11.zip` files. Unzip the files and change the path in `examples/amsterdam.py`. 
We borrowed the object detection results from BlazeIt and can be downloaded at [here](https://drive.google.com/drive/folders/1V6dJjo1JMM5QZbwoSyiLskDqZRxvcn3e?usp=share_link)

1.TASTI pretraining (Optional)

Follow [TASTI](https://github.com/stanford-futuredata/tasti) to generate train embedding NN with triplet loss and place the weight in `cache/tasti.pt`.
If you want to skip this step temporarily, you can modify the `config.index_model=None` (`examples/amsterdam.py`).

2.Run query script
```bash
python QaVA/examples/amsterdam.py
```

3.Run query-driven pattern selection
```bash
python QaVA/examples/auto_pattern.py
```