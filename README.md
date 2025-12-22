# حدیث‌یاب هوشمند

یک برنامه جستجوی معنایی که به کاربران امکان می‌دهد با وارد کردن جملات طبیعی، احادیث مرتبط را پیدا کنند. این برنامه از embedding سرویس Jina AI برای بازیابی نتایج معنایی مشابه از یک پایگاه داده برداری سرویس بک اند آماده Supabase استفاده می‌کند.

## دموی زنده

می‌توانید برنامه را به صورت زنده در آدرس زیر امتحان کنید:  
[https://hadithyab.com](https://hadithyab.onrender.com/)

## نمای کلی فنی

- **Frontend:** HTML, CSS, JS (از طریق قالب‌های Jinja2)
- **Backend:** Python Flask
- **مدل Embedding:** [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3)
- **مجموعه داده:** [`IslamShia/shia-hadith`](https://github.com/IslamShia/shia-hadith)
- **ذخیره‌ساز برداری:** Supabase + کتابخانه `vecs`

## ویژگی‌ها

- رابط جستجوی تحت وب ساخته شده با Flask و Jinja2
- تولید embedding از طریق Jina AI یا Hugging Face Inference API
- جستجوی شباهت با استفاده از کتابخانه `vecs` و ذخیره‌ساز برداری Supabase
- بازگرداندن متن عربی، ترجمه فارسی، منبع و اطلاعات راوی

## مدل Embedding

این برنامه از مدل `jina-embeddings-v3` برای تولید نمایش‌های برداری معنایی از احادیث استفاده می‌کند. این embeddings از ترجمه‌های فارسی احادیث برای پایگاه داده برداری تولید می‌شوند.

شما می‌توانید از مدل به دو روش استفاده کنید:

- **Jina AI Embeddings:** دسترسی به سرویس embedding Jina AI (پیش‌فرض: `jina-embeddings-v3`) برای embeddings با کیفیت بالا و ۱۰۲۴ بعدی از طریق API.
- **Hugging Face Embeddings:** استفاده از مدل از طریق Hugging Face Inference API با کلید API خودتان. اگر استنتاج در صفحه رسمی Jina AI غیرفعال باشد، می‌توانید از مدل سازگار میزبانی شده در [Sajjad313/my-Jira-embedding-v3](https://huggingface.co/Sajjad313/my-Jira-embedding-v3) استفاده کنید.

## پیش‌نیازها

- Python 3.8 یا جدیدتر
- یک محیط مجازی (توصیه می‌شود)
- یک حساب Supabase با یک ذخیره‌ساز برداری راه‌اندازی شده
- یک فایل `.env` (اختیاری) برای ذخیره متغیرهای محیطی به صورت محلی
- متغیرهای محیطی:
  - `CONNECTION_STRING`: URI اتصال برای ذخیره‌ساز برداری Supabase شما (استفاده شده توسط جستجو)
  - `JINA_API_KEY`: کلید API برای سرویس embedding Jina AI
  - `HF_API_KEY`: کلید API برای Hugging Face Inference API
  - `COLLECTION_NAME` و `NUM_RESULTS` (اختیاری) برای بازنویسی نام مجموعه جستجوی پیش‌فرض و تعداد نتایج

## نصب

1. مخزن را کلون کنید یا کد منبع را دانلود کنید:

   ```bash
   git clone <repository_url>
   cd production
   ```

2. یک محیط مجازی ایجاد و فعال کنید:

   ```bash
   python -m venv venv
   # در Windows:
   venv\Scripts\activate
   # در Unix یا macOS:
   source venv/bin/activate
   ```

3. بسته‌های پایتون مورد نیاز را نصب کنید:

   ```bash
   pip install -r requirements.txt
   ```

4. متغیرهای محیطی را تنظیم کنید (با مقادیر خودتان جایگزین کنید):

   ```bash
   set CONNECTION_STRING="your_connection_string"
   set JINA_API_KEY="your_jina_api_key"
   set HF_API_KEY="your_hf_api_key"
   ```

   در Unix/macOS، به جای `set` از `export` استفاده کنید.

## استفاده

1. سرور Flask را اجرا کنید:

   app.run() را در انتهای فایل `flask_server.py` از حالت کامنت خارج کنید

   سپس این دستور را در ترمینال اجرا کنید:

   ```bash
   python flask_server.py
   ```

2. مرورگر خود را باز کنید و به `http://127.0.0.1:5000` بروید.
3. یک جستجو در کادر ورودی وارد کنید و ارسال کنید تا احادیث مشابه برتر را ببینید.

## ساختار پروژه

```
.
├── config.py              # پیکربندی برنامه و بارگذاری متغیرهای محیطی
├── flask_server.py        # نقطه ورودی برنامه Flask
├── madules.py             # ابزارهای embedding و جستجوی شباهت
├── dev_maduels.py         # ابزارهای embedding و upsert دسته‌ای (استفاده توسعه‌دهنده)
├── requirements.txt       # وابستگی‌های پایتون
├── .env                   # بازنویسی‌های محلی متغیرهای محیطی (اختیاری)
├── templates/
│   └── index.html         # قالب Jinja2 برای رابط کاربری جستجو
└── README.md              # نمای کلی پروژه و دستورالعمل‌ها
```

## استقرار در محیط تولید

این برنامه آماده استقرار در محیط تولید پشت یک سرور WSGI مانند **Waitress** یا **Gunicorn** است.

- در Windows، از Waitress استفاده کنید (از طریق requirements.txt نصب شده):

  ```powershell
  waitress-serve --listen=0.0.0.0:5000 flask_server:app
  ```

