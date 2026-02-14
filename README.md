# Contextualized Visual Personalization in Vision-Language Models

* **Authors**: Yeongtak Oh*, Sangwon Yu*, Junsung Park, Han Cheol Moon, Jisoo Mok, Sungroh Yoon
(*: Equal contribution)

[![arXiv](https://img.shields.io/badge/arXiv-2602.03454-b31b1b.svg)](https://arxiv.org/abs/2602.03454) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CoViP_Model-yellow)](https://huggingface.co/Yeongtak/CoViP-Qwen3-VL-8B-GSPO) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Full_Dataset-yellow)](https://huggingface.co/datasets/Yeongtak/benchmark_person_pmmlm_v2) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Person_Dataset-yellow)](https://huggingface.co/datasets/Yeongtak/benchmark_CoViP_captioning) 

[![Google Drive](https://img.shields.io/badge/Google%20Drive-Test_Dataset-blue)](https://drive.google.com/file/d/1Ma2g1oSzl8ya0A-wJXMhPHekGLPqlvnW/view?usp=sharing) 

<p align="center">
  <img src="./imgs/figure1.png" alt="Figure 1. Qualitative example of the use-case for contextual visual personalization in VLMs. Note that our CoViP effectively responds to the question while integrating the mentioned personal details from the given multimodal contexts." width="100%">
</p>


---

## 📝 Abstract

Despite recent progress in vision-language models (VLMs), existing approaches often fail to generate personalized responses based on the user's specific experiences, as they lack the ability to associate visual inputs with a user's accumulated visual-textual context. We newly formalize this challenge as **contextualized visual personalization**, which requires the visual recognition and textual retrieval of personalized visual experiences by VLMs when interpreting new images. 

To address this issue, we propose **CoViP**, a unified framework that treats personalized image captioning as a core task for contextualized visual personalization and improves this capability through reinforcement-learning-based post-training and caption-augmented generation. We further introduce diagnostic evaluations that explicitly rule out textual shortcut solutions and verify whether VLMs truly leverage visual context. Extensive experiments demonstrate that existing open-source and proprietary VLMs exhibit substantial limitations, while CoViP not only improves personalized image captioning but also yields holistic gains across downstream personalization tasks. These results highlight CoViP as a crucial stage for enabling robust and generalizable contextualized visual personalization.

---

## 📅 Do-lists
- [x] We released evaluation codes for personalized image captioning!
- [x] We released training codes for CoViP!

---

## 🔧 Training

To train the model, please follow the setup and configuration used in the **`VLM-R1-Qwen3`** branch.

### 📦 Training Data

The following data files are required for training:

- **JSON configuration file** [![Google Drive](https://img.shields.io/badge/Google%20Drive-JSON-blue)](https://drive.google.com/file/d/1gdH8HnkIra5P_eFoMuBDaqqVu9jM72V4/view?usp=sharing) 

- **Training dataset file** [![Google Drive](https://img.shields.io/badge/Google%20Drive-Training_data-blue)](https://drive.google.com/file/d/1Vgl3vSQXHzgB9sTW03H7KQxJk8a8peln/view?usp=sharing) 
  

### ⚙️ Data Configuration

1. Download the JSON file and place it in your local data directory.
2. Update the dataset path by adding the appropriate prefix in the following configuration file:

```bash
VLM-R1-Qwen3/src/open-r1-multimodal/data_config/pmllm.yaml
```
Ensure that the paths correctly point to the downloaded JSON and training data files.


Training is performed using on-policy reinforcement learning algorithms.

  1.	Navigate to the run script directory:

  ```bash
  VLM-R1-Qwen3/src/open-r1-multimodal/run_scripts
  ```

  2.	Select and execute the training script corresponding to your desired on-policy RL algorithm (e.g., GRPO, DrGRPO, GSPO).

Each script is pre-configured for a specific training setup; adjust hyperparameters as needed before execution.

📌 Notes
	•	Make sure the environment setup follows the instructions provided in the VLM-R1-Qwen3 branch.
	•	Training scripts assume that the dataset configuration has been correctly updated in pmllm.yaml.

---

## 🚀 Inference Example

### 1. Caption Generation
Use the following notebook to generate captions with CoViP on the test benchmark:
- `generate_caption_qwen.ipynb`

### 2. Caption Evaluation & vLLM Porting
After generating the captions, execute the porting script for the vLLM environment:
```bash
# Execute with localhost
./evaluation/vllm_porting.sh
```
  (Run with `localhost`)
 
### 3. **Evaluate captions with LLM-as-a-Judge**  
   Execute `CapEval_QAs_save.py` to score the generated captions using MCQA-based evaluation.

---

## 📦 Dataset Configuration

We release the train and test splits of the personalized image captioning benchmark used for training CoViP.

In addition, to support human-centric personalization scenarios, we provide an additional person-only benchmark dataset.

* **Hugging Face Dataset Repository Names**
    * **1) Personalized Image Captioning**: 🔗 [Yeongtak/benchmark_CoViP_captioning](https://huggingface.co/datasets/Yeongtak/benchmark_CoViP_captioning)
      
    * **2) Person-only Personalized Image Captioning**: 🔗 [Yeongtak/benchmark_person_pmmlm_v2](https://huggingface.co/datasets/Yeongtak/benchmark_person_pmmlm_v2)


✨ Both datasets are intended for research purposes, and you are very welcome to use them in your work.
If you have questions or suggestions, feel free to open an issue!

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{oh2026contextualized,
  title={Contextualized Visual Personalization in Vision-Language Models},
  author={Oh, Yeongtak and Yu, Sangwon and Park, Junsung and Moon, Han Cheol and Mok, Jisoo and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2602.03454},
  year={2026}
}
```

---

## Acknowledgements

We gratefully acknowledge the following open-source repositories and resources that supported our work:

- https://github.com/oyt9306/RePIC
- https://github.com/QwenLM/Qwen3-VL
