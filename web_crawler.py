import pyautogui

sw, sh = pyautogui.size()
print(sw,sh)
cmx,cmy = pyautogui.position()
print(cmx,cmy)


pyautogui.moveTo(50,50)
pyautogui.move(300,300,5)