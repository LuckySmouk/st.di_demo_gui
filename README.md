# st.di_demo_gui

sd_demo_gpu.py # Исполняемы файл

Требования:

Видеокарта: Geforce 30x-40x
ОС: 		Windows 10/11
CUDA 		12.1
Microsoft ???
Python 		3.10-3.11


скачать модель
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1

в папке ./bin/models должна находиться модель stable-diffusion-2-1

Если уже скачана другая модель изменить внутри исполняемого файла путь к модели.
В этой переменной путь к модели 'model_path = "bin/model/stable-diffusion-2-1"'

Можно скачать любую другую по ссылке во вкладке Models
https://huggingface.co/stabilityai

Запуск:
1. В терминале перейти в директорию с исполняемым файлом "cd ВашПуть/sd_demo/bin"
2. Ввести "pip install --no-cache-dir -r requirements.txt" # Установка необходимых библиотек
3. Ввести "streamlit run sd_demo_gpu.py" # Запуск локального интерфейса в браузере
