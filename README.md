pip install -r requirements.txt

python training/train_crop_model.py
python training/train_fertilizer_model.py

streamlit run app.py
