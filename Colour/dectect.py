import cv2
import numpy as np
import pandas as pd
import argparse

df = pd.read_csv('color_names.csv', encoding='utf-8')
clicked = False

# # Apply hex_to_rgb to each row and expand the result into three columns
# def hex_to_rgb(hex_code):
#     hex_code = hex_code.lstrip('#')
#     return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
#
# df[['R', 'G', 'B']] = df['hex'].apply(hex_to_rgb).apply(pd.Series)
# df['color'] = df['name'].str.lower().str.replace(" ", "_")
# df = df[['name', 'color', 'hex', 'R', 'G', 'B']]
# df.to_csv('color_names.csv', index=False)

# Taking an image from the user
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Image Path')
args = vars(ap.parse_args())
img_path = args['image']

# Set a mouse callback event on a window
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# Reading image with opencv
img = cv2.imread(img_path)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)

# Calculate distance to get color name
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"]))+ abs(B - int(df.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            name = df.loc[i, "name"]
    return name

# Display image on the window
while (1):
    cv2.imshow("image",img)
    if (clicked):
        # cv2.rectangle(image, startpoint, endpoint, color, thickness) -1 thickness fills rectangle entirely
        cv2.rectangle(img, (20, 20), (img.shape[1] - 20, 60), (b, g, r), -1)
        # Creating text string to display (Color name and RGB values)
        text = getColorName(r, g, b) + ': R = '+ str(r) + ' G = '+ str(g) + ' B = '+ str(b)
        # cv2.putText(img, text, start, font(0-7), fontScale, color, thickness, lineType, (optional bottomLeft bool))
        cv2.putText(img, text,(50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if (r + g + b >= 600): cv2.putText(img, text,(50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        clicked = False

    # Break the loop when user hits 'esc' key
    if (cv2.waitKey(20) & 0xFF == 27) | (cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1): break

cv2.destroyAllWindows()