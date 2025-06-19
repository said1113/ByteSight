import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import numpy as np
import aiohttp

# load envioremental variables
load_dotenv()

# AI Model paths
MODEL_BASE_DIR_FROM_ENV = os.getenv('MODEL_BASE_DIRECTORY')

# envioremental variables check if loaded correctly
if MODEL_BASE_DIR_FROM_ENV is None:
    print("WARNING: MODEL_BASE_DIRECTORY environment variable not set in .env file.")
    print("Falling back to a relative path. Please ensure your 'converted_keras (1)' folder is in the script's directory.")
    # Fallback to a relative path if the env var isn't found
    MODEL_BASE_DIR = 'converted_keras (1)'
else:
    MODEL_BASE_DIR = MODEL_BASE_DIR_FROM_ENV

# full paths
MODEL_PATH = os.path.join(MODEL_BASE_DIR, 'keras_model.h5')
LABELS_PATH = os.path.join(MODEL_BASE_DIR, 'labels.txt')


# discord bot initialize
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents, help_command=None) # Ensure help_command=None is here

#PC_part info
PC_PART_INFO = {
    "CPU": (
        "**CPU, Merkezi İşlem Birimi (Central Processing Unit) anlamına gelir.**\n"
        "Bilgisayarın **beyni** olarak da bilinen bu birim, bir bilgisayar sistemindeki temel hesaplama işlemlerini gerçekleştiren donanım bileşenidir.\n"
        "CPU, talimatları işler, verileri işler ve bilgisayarın diğer tüm bileşenlerinin koordineli bir şekilde çalışmasını sağlar."
    ),
    "GPU": (
        "**GPU, Grafik İşlem Birimi (Graphics Processing Unit) anlamına gelir.**\n"
        "Özellikle grafik ve görsel hesaplamalar için tasarlanmıştır. Oyunlar, video düzenleme ve 3D modelleme gibi grafik ağırlıklı işlerde kullanılır.\n"
        "Görüntüleri hızlı bir şekilde işleyerek ekranınızda görmenizi sağlar."
    ),
    "RAM": (
        "**RAM, Rastgele Erişimli Bellek (Random Access Memory) anlamına gelir.**\n"
        "Bilgisayarın o an aktif olarak kullandığı verileri geçici olarak depolayan hızlı bir bellek türüdür.\n"
        "Uygulamaların hızlı çalışmasını ve çoklu görev yapabilmeyi sağlar. Bilgisayar kapatıldığında içindeki veriler silinir."
    ),
    "MOTHERBOARD": (
        "**Anakart (Motherboard), bilgisayarın tüm temel bileşenlerini birbirine bağlayan ana devre kartıdır.**\n"
        "CPU, RAM, GPU, depolama aygıtları ve diğer çevre birimlerinin iletişim kurmasını sağlar.\n"
        "Bilgisayarın omurgası gibidir."
    ),
    "SATA SSD": (
        "**SATA SSD, Serial ATA arayüzünü kullanan bir Katı Hal Sürücüsüdür (Solid State Drive).**\n"
        "Geleneksel HDD'lere göre çok daha hızlı veri okuma/yazma hızları sunar ve hareketli parçaları olmadığı için daha dayanıklıdır.\n"
        "Çoğu masaüstü ve dizüstü bilgisayarda depolama çözümü olarak kullanılır."
    ),
    "NVME SSD": (
        "**NVMe SSD, Non-Volatile Memory Express arayüzünü kullanan bir Katı Hal Sürücüsüdür (Solid State Drive).**\n"
        "SATA SSD'lerden çok daha yüksek hızlar sunar çünkü doğrudan PCIe slotlarına takılarak anakart ile daha hızlı iletişim kurar.\n"
        "Yüksek performans gerektiren uygulamalar ve oyunlar için idealdir."
    ),
    "HDD": (
        "**HDD, Sabit Disk Sürücüsü (Hard Disk Drive) anlamına gelir.**\n"
        "Verileri manyetik plakalar üzerinde depolayan geleneksel bir depolama aygıtıdır. Yüksek kapasiteleri uygun maliyetle sunar.\n"
        "SSD'lere göre daha yavaştır ve hareketli parçaları olduğu için darbelere karşı daha hassastır."
    ),
    "PSU": (
        "**PSU, Güç Kaynağı Birimi (Power Supply Unit) anlamına gelir.**\n"
        "Bilgisayarın tüm bileşenlerine doğru voltajda ve miktarda elektrik gücü sağlayan donanım bileşenidir.\n"
        "Bilgisayarın kararlı çalışması için hayati öneme sahiptir."
    ),
    "AIR COOLING": (
        "**Hava Soğutma (Air Cooling), bilgisayar bileşenlerini, özellikle CPU'yu, ısıyı dağıtmak için hava akımını kullanan bir soğutma yöntemidir.**\n"
        "Genellikle bir soğutucu blok (heat sink) ve bir veya daha fazla fan içerir. Isıyı bileşenden alıp havaya aktarır."
    )
}

# tensorflow
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Successfully loaded model from: {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load model from {MODEL_PATH}. Please check the path and file integrity.")
    print(f"Details: {e}")
    exit() # Exit the script if the model cannot be loaded

# labels
try:
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f]
    print(f"Successfully loaded labels from: {LABELS_PATH}")
except FileNotFoundError:
    print(f"WARNING: Labels file not found at {LABELS_PATH}. Using default labels.")
    labels = ['CPU', 'GPU', 'RAM', 'Motherboard', 'Sata SSD', 'NVMe SSD', 'HDD', 'PSU', 'Air Cooling']
except Exception as e:
    print(f"WARNING: Error reading labels file from {LABELS_PATH}: {e}. Using default labels.")
    labels = ['CPU', 'GPU', 'RAM', 'Motherboard', 'Sata SSD', 'NVMe SSD', 'HDD', 'PSU', 'Air Cooling']


# bot introduces it self
@bot.event
async def on_ready():
    print(f'{bot.user} olarak giriş yaptık')


@bot.command()
async def hello(ctx):
    await ctx.send(f'Merhaba! Ben {bot.user}, bir Discord PC parça botuyum!')

@bot.command()
async def easter_egg(ctx):
    await ctx.send(f'Hey {bot.user}, bugün bir kod yok. Sonra tekrar dene!')

#help
@bot.command()
async def help(ctx):
    help_message = (
        f'Aşağıdaki komut ve sintakslar **ByteSight** discord botu için özelleştirilmiştir!\n\n'
        f'**Komutlar:**\n'
        f'`$hello` - Bot kendini tanıtır.\n'
        f'`$predict` - Yüklediğiniz bir fotoğrafın içindeki PC parçasını tahmin eder. (Fotoğrafı komutla birlikte ekleyin)\n'
        f'`$net_predict <resim_url>` - Belirttiğiniz URL adresindeki fotoğrafın içindeki PC parçasını tahmin eder.\n'
        f'`$easter_egg` - Küçük bir sürpriz mesaj gönderir.\n'
        f'`$info <parça ismi>` - Bir PC Parçası ile ilgili bilgi verir.\n\n '
        f'**Kullanım Örnekleri:**\n'
        f'`$predict` (ve bir resim ekleyin)\n'
        f'`$net_predict https://example.com/some_pc_part.jpg`\n'
        f'`$info (parça ismi)` - Örneğin - `$info CPU` \n\n'
        f'**Kullanırken dikkat edin! Gönderilen fotoğraflarda nesnenin resimde yalnız olduğundan emin olun. Aksi taktirde yanlış cevap verme olasılığı daha yüksek!**\n\n'
        f'Neleri tanıya bilirsiniz:\n'
        f'0 CPU (Central Processing Unit)\n'
        f'1 GPU (Graphics Processing Unit)\n'
        f'2 RAM (Random Access Memory)\n'
        f'3 Motherboard\n'
        f'4 Sata SSD\n'
        f'5 NVMe SSD\n'
        f'6 HDD\n'
        f'7 PSU (Power Supply)\n'
        f'8 Air Cooling (Hava Soyutma)\n\n'
        f'**Bu discord botu AI kullanmaktatır. Yanlış cevap vere bilir!**'
    )
    await ctx.send(help_message)


# image convertion to useable state
def preprocess_image(image_bytes, target_size):
    img = Image.open(io.BytesIO(image_bytes)).resize(target_size)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# prediction
async def predict_image(image_bytes):
    try:
        img_array = preprocess_image(image_bytes, (model.input_shape[1], model.input_shape[2]))
        predictions = model.predict(img_array)
        decoded_predictions = tf.nn.softmax(predictions).numpy()
        top_prediction_index = np.argmax(decoded_predictions[0])
        confidence = decoded_predictions[0][top_prediction_index]
        predicted_label = labels[top_prediction_index]
        return predicted_label, confidence

    except Exception as e:
        print(f"Fotoğrafı işlemek mümkün olmadı: {e}")
        return None, None

# Event to handle messages with attachments
@bot.command()
async def predict(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_bytes = await attachment.read()
                prediction, confidence = await predict_image(image_bytes)

                if prediction:
                    await ctx.send(f"Bu bir **{prediction}** ve ben bunu **{confidence:.2f}%** güvenle söylüyorum.")
                else:
                    await ctx.send("Fotoğrafı işlemek mümkün olmadı.")
            else:
                await ctx.send("Lütfen bir fotoğraf dosyası eklediğinizden emin olun.")
    else:
        await ctx.send("Lütfen tahmin etmek için bir fotoğraf ekleyin.")

# url prediction function
@bot.command()
async def net_predict(ctx, image_url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    image_bytes = await resp.read()
                    prediction, confidence = await predict_image(image_bytes)
                    if prediction:
                        await ctx.send(f"Sitedeki fotoğraf bir **{prediction}** gibi gözüküyor. Ve ben bunu **{confidence:.2f}%** güvenle söylüyorum.")
                    else:
                        await ctx.send("Sitedeki fotoğraf işlenemedi.")
                else:
                    await ctx.send(f"Sitedeki fotoğrafı yüklemek mümkün olmadı. Durum kodu: {resp.status}")
    except aiohttp.ClientConnectorError as e:
        await ctx.send(f"Belirtilen URL'ye bağlanılamadı: {e}. Lütfen URL'nin doğru olduğundan emin olun.")
    except Exception as e:
        await ctx.send(f"Bir problem oldu: {e}")

# basic info function
@bot.command()
async def info(ctx, *, part_name: str):

    part_name_upper = part_name.upper()

    if "AIR COOLING" in part_name_upper or "HAVA SOĞUTMA" in part_name_upper:
        requested_part = "AIR COOLING"
    elif "SATA SSD" in part_name_upper:
        requested_part = "SATA SSD"
    elif "NVME SSD" in part_name_upper:
        requested_part = "NVME SSD"
    elif "HDD" in part_name_upper:
        requested_part = "HDD"
    elif "PSU" in part_name_upper:
        requested_part = "PSU"
    elif "RAM" in part_name_upper:
        requested_part = "RAM"
    elif "GPU" in part_name_upper or "EKRAN KARTI" in part_name_upper:
        requested_part = "GPU"
    elif "CPU" in part_name_upper or "İŞLEMCİ" in part_name_upper:
        requested_part = "CPU"
    elif "MOTHERBOARD" in part_name_upper or "ANAKART" in part_name_upper:
        requested_part = "MOTHERBOARD"
    else:
        requested_part = part_name_upper

    info_message = PC_PART_INFO.get(requested_part)

    if info_message:
        await ctx.send(info_message)
    else:
        await ctx.send(
            f"Üzgünüm, **{part_name}** hakkında bilgi bulamadım. "
            "Şu an için bilgi sağlayabildiğim parçalar: CPU, GPU, RAM, Motherboard, SATA SSD, NVMe SSD, HDD, PSU, Air Cooling."
            "Lütfen listedeki isimleri kullanmaya çalışın."
        )



#bot token - retrieve from environment variable
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
if DISCORD_TOKEN is None:
    print("ERROR: DISCORD_BOT_TOKEN environment variable not set.")
    print("Please create a .env file in the same directory as your script with DISCORD_BOT_TOKEN=YOUR_TOKEN_HERE")
    exit()

bot.run(DISCORD_TOKEN)