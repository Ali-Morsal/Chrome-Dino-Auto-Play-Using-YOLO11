from PIL import Image, ImageOps

# مسیر تصویر ورودی
input_path = "DinoRun2.png"
# مسیر تصویر خروجی
output_path = "negative.png"

# باز کردن تصویر با حفظ آلفا
img = Image.open(input_path).convert("RGBA")

# جداسازی کانال‌ها
r, g, b, a = img.split()

# نگاتیو کردن کانال RGB
rgb_image = Image.merge("RGB", (r, g, b))
neg_rgb = ImageOps.invert(rgb_image)

# ترکیب دوباره با کانال آلفا
neg_image = Image.merge("RGBA", (*neg_rgb.split(), a))

# ذخیره
neg_image.save(output_path)

print(f"تصویر نگاتیو با حفظ آلفا ذخیره شد: {output_path}")
