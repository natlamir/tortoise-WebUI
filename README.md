Check out original repo for more details and usage: [Original Repo](https://github.com/neonbjb/tortoise-tts)

# Windows Install
1. Create and activate new conda environment.
```
conda create --name tortoise-WebUI python=3.9 -y && conda activate tortoise-WebUI
```

2. Install pytorch.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

3. Confirm CUDA is available
```
python
import torch
torch.cuda.is_available()
exit()

```

4. Clone the repository and cd into the new folder
```
git clone https://github.com/natlamir/tortoise-WebUI.git && cd tortoise-WebUI
```

5. Install dependencies
```
pip install -r requirements.txt
```

6. Install pysoundfile
```
conda install -c conda-forge pysoundfile -y
```

7. Run tortoise setup file
```
python setup.py install
```

# Usage
Using the example command from original repo
```
python tortoise/do_tts.py --text "we have now re-installed tortoise. enjoy!" --voice random --preset fast
```

Or use this command to locally run the gradio web UI (same as the one from [huggingface space](https://huggingface.co/spaces/Manmay/tortoise-tts)) with some modifications to be able to use locally
```
python app.py
```

