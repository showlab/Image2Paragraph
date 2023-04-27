# buat variable a dan b isi masing masing nilai 10 dan 20
a = 10
b = 20

# jumlahkan nilai a dan b simpan dalam variable c
c = a + b
# cetak nilai c
print(c)


# buat kelas untuk menghitung luas segitiga

class Segitiga:
    def __init__(self, alas, tinggi):
        self.alas = alas
        self.tinggi = tinggi

    def luas(self):
        return self.alas * self.tinggi / 2


# buat variable yang menampung input alas dan tinggi dari user
alas = int(input("Masukkan alas: "))
tinggi = int(input("Masukkan tinggi: "))

# buat variable segitiga yang menampung hasil dari kelas Segitiga
segitiga = Segitiga(alas, tinggi)
# cetak hasil dari luas segitiga dengan awal kalimat Luas segitiga adalah
print("Luas segitiga adalah: ")
print(segitiga.luas())
