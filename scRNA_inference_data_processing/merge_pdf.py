import os
import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Spacer, PageBreak
from reportlab.pdfgen import canvas

# --- НАСТРОЙКИ ---
IMG_DIR = "/mnt/jack-5/amismailov/miRNA_study/supplementary_plots"
OUTPUT_PDF = "/mnt/jack-5/amismailov/miRNA_study/supplementary_overview_121_datasets.pdf"
IMAGES_PER_PAGE = 3

def natural_sort_key(s):
    """ Ключ для сортировки, чтобы RCC_sample_2 шёл перед RCC_sample_10 """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class NumberedCanvas(canvas.Canvas):
    """ Кастомный канвас для автоматического добавления номеров страниц """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        # Текст внизу страницы по центру
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawCentredString(A4[0] / 2.0, 30, text)
        
        # Небольшая линия над номером страницы для академического вида
        self.setLineWidth(0.5)
        self.setStrokeColorRGB(0.7, 0.7, 0.7)
        self.line(50, 45, A4[0] - 50, 45)

def build_pdf():
    # 1. Собираем и сортируем все PNG файлы
    if not os.path.exists(IMG_DIR):
        print(f"Ошибка: Директория {IMG_DIR} не найдена!")
        return

    files = [f for f in os.listdir(IMG_DIR) if f.endswith('_report.png')]
    files.sort(key=natural_sort_key)
    
    print(f"Найдено {len(files)} рисунков для сборки.")

    if len(files) == 0:
        print("Нечего собирать. Завершение работы.")
        return

    # 2. Настраиваем геометрию А4 страницы (размеры в пунктах: 595.27 x 841.89)
    # Делаем отступы поменьше (по 0.4 дюйма), чтобы максимизировать место под графики
    margin = 0.4 * inch
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin + 20 # чуть больше снизу для номера страницы
    )

    # Вычисляем доступную ширину и высоту для картинок
    available_width = A4[0] - (2 * margin)
    available_height = A4[1] - (2 * margin) - 20
    
    # Высота одной картинки при размещении 3-х штук на страницу (с учетом небольших зазоров)
    img_width = available_width
    img_height = (available_height - (2 * 15)) / IMAGES_PER_PAGE 

    story = []

    # 3. Наполняем PDF элементами
    for index, filename in enumerate(files):
        img_path = os.path.join(IMG_DIR, filename)
        
        # Создаем объект картинки с жестко заданными размерами (пропорции сохраняются)
        img = Image(img_path, width=img_width, height=img_height)
        story.append(img)
        
        # Определяем, что делать дальше
        is_last_image = (index == len(files) - 1)
        is_page_end = ((index + 1) % IMAGES_PER_PAGE == 0)

        if not is_last_image:
            if is_page_end:
                story.append(PageBreak()) # Переходим на новую страницу
            else:
                story.append(Spacer(1, 15)) # Отступ между графиками на одной странице

    # 4. Сборка документа
    print("Рендеринг PDF... Это может занять около минуты для 121 тяжелого PNG.")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Успешно! Финальный файл сохранен в: {OUTPUT_PDF}")

if __name__ == "__main__":
    build_pdf()