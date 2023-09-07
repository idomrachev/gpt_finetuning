import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelWithLMHead.from_pretrained('./checkpoint1300model07.09')


import telepot
from telepot.loop import MessageLoop
TOKEN = '6085029622:AAGgHyAlLB68fcVpuk9tvnKYK4zo0LjItus'
# Функция для обработки входящих сообщений
def handle_message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    text1='привет'
    text2='привет'
    # Проверяем тип сообщения (текстовое сообщение)
    if content_type == 'text':
        # Получаем текст сообщения
        text = msg['text']
        inputs = tokenizer('@@ПЕРВЫЙ@@ привет @@ВТОРОЙ@@ привет @@ПЕРВЫЙ@@ как дела? @@ВТОРОЙ@@', return_tensors='pt')

        text = f" @@ПЕРВЫЙ@@ {text2} @@ВТОРОЙ@@ {text1} @@ПЕРВЫЙ@@ {text} @@ВТОРОЙ@@"
        inputs = tokenizer(text, return_tensors='pt')
        generated_token_ids = model.generate(
            **inputs,
            top_k=30,
            top_p=0.95,
            num_beams=3,
            num_return_sequences=1,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=2.2,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40
        )

        context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
        context_with_response1 = context_with_response[0].split('@@ВТОРОЙ@@')[-1]

        bot.sendMessage(chat_id, context_with_response1)
        text2=text1
        text1=text


# Создаем экземпляр бота
bot = telepot.Bot(TOKEN)

# Настраиваем обработчик входящих сообщений
MessageLoop(bot, handle_message).run_as_thread()

print('Бот запущен. Готов к приему сообщений.')

# Бесконечный цикл для работы бота
while True:
    pass






