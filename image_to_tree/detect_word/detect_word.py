



def flood_fill()

def find_char_postion(img, find_x,find_y,):
    

    return 0

def detect_word(img):
    w,h,c = img.shape()
    default_char_wh_ratio = 1
    wait_ratio = 0.4
    watch_range = 0.6

    sx, sy = 0, int((1-watch_range)/2*h)
    for x in range(sx, int(wait_ratio*h)):
        for y in range(sy, sy + int(watch_range*h)):
            if img[y,x] == 255:
                find_char_postion(img,x,y,sx,sy,default_char_wh_ratio)
    find_char_positon()




    return 0





if __name__ == "__main__":
    img =
    detect_word(img)