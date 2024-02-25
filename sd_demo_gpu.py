import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Проверка доступности CUDA
print(torch.cuda.is_available())  


st.set_page_config(page_title="SD Demo")

# Загрузка модели
model_path = "model/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.steps = 25
pipe.scheduler.num_train_timesteps = 1000
pipe.scheduler.learning_rate = 0.07
pipe.scheduler.batch_size = 16
pipe = pipe.to("cuda")

# Создание боковой панели для промптов и параметров
st.sidebar.header('Введите промпт и параметры')
prompt = st.sidebar.text_input("Промпт:")  
negative_prompt = st.sidebar.text_input("Отрицательный промпт:")
alpha = st.sidebar.slider("Alpha (0.0 - 1.0)", 0.0, 1.0, 0.5)
beta = st.sidebar.slider("Beta (0.0 - 1.0)", 0.0, 1.0, 0.5)
temperature = st.sidebar.slider("Температура (0.1 - 1.0)", 0.1, 1.0, 0.5)
pipe.alpha = alpha
pipe.beta = beta
pipe.temperature = temperature

# Создание кнопки для генерации изображения
if st.button('Сгенерировать'):
    with st.spinner('Генерация...'):
        image = pipe(prompt, negative_prompts=negative_prompt).images[0]
    
    # Отображение изображения  
    st.image(image)

if __name__ == '__main__':
    # Запуск приложения
    st.markdown("# Демонстрация Stable Diffusion")
    st.write("Введите промпты и параметры, затем нажмите 'Сгенерировать'!")