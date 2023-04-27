import openai


class ImageToText:
    def __init__(self, api_key, gpt_version="gpt-3.5-turbo"):
        self.template = self.initialize_template()
        openai.api_key = api_key
        self.gpt_version = gpt_version

    # def initialize_template(self):
    #     prompt_prefix_1 = """Generate only an informative and nature paragraph based on the given information(a,b,c,
    #     d):\n"""



    def initialize_template(self):
        prompt_prefix_1 = """Hasilkan hanya paragraf informatif dan alamiah berdasarkan informasi yang diberikan(
        b, c, d):\n"""
       # prompt_prefix_2 = """\n a. Resolusi Gambar:  """
        prompt_prefix_3 = """\n b. Keterangan Gambar: """
        prompt_prefix_4 = """\n c. Keterangan Padat: """
        prompt_prefix_5 = """\n d. Semantik Wilayah: """
        prompt_suffix = """\n Ada beberapa aturan:
        Buatlah paragraf informatif dan ringkas yang mendeskripsikan konten foto, berdasarkan aturan-aturan berikut ini:
        1. Berikan deskripsi singkat dan informatif tentang konten foto, termasuk apa yang terjadi, subjek utama, dan fitur menonjol lainnya.
        2. Sertakan kata kunci yang relevan untuk foto, meliputi subjek, objek, warna, tema, dan fitur penting lainnya.
        3. Tunjukkan objek, warna (jika bukan foto hitam putih), dan posisi dengan menggunakan kata benda daripada koordinat untuk informasi posisi setiap objek.
        4. Batasi deskripsi menjadi tidak lebih dari 7 kalimat.
        5. Sajikan deskripsi konten dalam satu paragraf.
        6. Gambarkan posisi relatif setiap objek dalam foto.
        7. Hindari penggunaan angka atau koordinat dalam deskripsi konten. Fokus pada penggunaan kata benda dan deskripsi visual untuk posisi objek.
        8. perkirakan kegiatan atau aktifitas yang terjadi pada foto tersebut.
        9. jika diketahui gambarkan konteks sejarah pada foto tersebut.
        Gunakan aturan-aturan ini untuk membuat deskripsi konten foto yang informatif, ringkas, dan mudah dibaca, sesuai dengan standar internasional.           
        """
        # template = f"{prompt_prefix_1}{prompt_prefix_2}{{width}}X{{height}}{prompt_prefix_3}{{caption}}{prompt_prefix_4}{{dense_caption}}{prompt_prefix_5}{{region_semantic}}{prompt_suffix}"
        template = f"{prompt_prefix_1}{prompt_prefix_3}{{caption}}{prompt_prefix_4}{{dense_caption}}{prompt_prefix_5}{{region_semantic}}{prompt_suffix}"
        return template

    # Show object, color and position.
    # Use nouns rather than coordinates to show position information of each object.
    # No more than 7 sentences.
    # Only use one paragraph.
    # Describe position of each object.
    # Do not appear number.

    def paragraph_summary_with_gpt(self, caption, dense_caption, region_semantic, width, height):
        question = self.template.format(width=width, height=height, caption=caption, dense_caption=dense_caption,
                                        region_semantic=region_semantic)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nStep4, Paragraph Summary with GPT-3:')
        print('\033[1;34m' + "Question:".ljust(10) + '\033[1;36m' + question + '\033[0m')
        completion = openai.ChatCompletion.create(
            model=self.gpt_version,
            messages=[
                {"role": "user", "content": question}]
        )

        print('\033[1;34m' + "ChatGPT Response:".ljust(18) + '\033[1;32m' + completion['choices'][0]['message'][
            'content'] + '\033[0m')
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return completion['choices'][0]['message']['content']

    def paragraph_summary_with_gpt_debug(self, caption, dense_caption, width, height):
        question = self.template.format(width=width, height=height, caption=caption, dense_caption=dense_caption)
        print("paragraph_summary_with_gpt_debug:")
        return question
