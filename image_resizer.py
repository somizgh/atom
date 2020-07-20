from PIL import Image



def image_resize(source,x,y):
    image = Image.open(source)
    resized = image.resize((x,y))
    return resized


def show_image(img):

    img.show()
    return 0

show_image(image_resize('./data/web_page_test.jpg',3820,2160))