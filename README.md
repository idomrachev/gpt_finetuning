# gpt_finetuning
Решение кейса по NLP для смены по машинному обучению от Тинькофф в Университете "Сириус"
Ход решения и промежуточные результаты оформлены в jupyter ноутбуке 'fine_tuning_gpt.ipynb
Код для запуска телеграм-бота лежит в файле gpt_tg_bot.ipynb
# Описание решения:
За основу взята модель с huggingface  'ruDialoGPT-medium': https://huggingface.co/tinkoff-ai/ruDialoGPT-medium
Данные были загружены из чата общежития ДАС МГУ. Основные темы общения - учеба, быт, общественная деятельность, юмор, развлечения.
Далее данные были приведены к нужному формату, токенизированы и собраны в датасет. Для обучения модели я взял 2500 последовательностей. 
Модель обучалась 3 часа, прошла 3 эпохи. Подробнее процесс обучения описан в ноутбуке: https://colab.research.google.com/drive/1ZmVJE-reOaztOw8RvOCUXn47XhAyQQAE?usp=sharing
На тестировании модель работает преимущественно корректно, если подбирать параметры, можно добиться интересных результатов. (Примеры работы в ноутбуке с обучением)

Получив модель, я сохранил ее на диск и написал телеграм бота на основе нее. Модель на гугл диске: https://drive.google.com/drive/folders/1KWkMg10dWjlUYEpY67TSWRSVTlxSOrjw?usp=sharing

Далее я сохранил скрипт запуска бота, модель и обернул в докер.

Модель работает в телеграм боте по ссылке: https://t.me/gpt2_tink_bot

