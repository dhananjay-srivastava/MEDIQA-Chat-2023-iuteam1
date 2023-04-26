# MEDIQA-Chat-2023-iuteam1

models available on hugging face at: https://huggingface.co/dhananjay2912

Plots and Hyperparameters available at

Section specific models:
https://api.wandb.ai/links/dsteam1/onxkm0mp

Section specific models (pubmed finetuned):
https://wandb.ai/dsteam1/LSG_BART_PUBMED/reports/Section-specific-model-report--Vmlldzo0MTc5NDg4?accessToken=zvp9qfgjwm1yfpe351zsld8gr0hw0fjpl4btxypqktv9p5aw6wuk7mzxk10a1ol2

Combiner model:
https://api.wandb.ai/links/dsteam1/juww1i71

Procedure to recreate results for MediQA Chat Shared Task B for Clinical Conversation Sumamarization

Place the test set file from the ACI Demo Dataset in the folder

For Run 1
./install.sh
./activate.sh
decode_taskB_run1.sh taskB_testset4participants_inputConversations.csv

For Run 2
./install.sh
./activate.sh
decode_taskB_run2.sh taskB_testset4participants_inputConversations.csv

For Run 3
./install.sh
./activate.sh
decode_taskB_run3.sh taskB_testset4participants_inputConversations.csv


Training Script:
TBD
