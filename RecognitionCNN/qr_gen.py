import qrcode
import os

path = os.path.dirname(os.path.abspath(__file__))

def generate_qr(data, filename):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(path + r'/QRs/qr-' + filename + ".png")
    print("QR-код сохранён как ", filename)