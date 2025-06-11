import cv2
import numpy

print(f"Versao do OpenCV: {cv2.__version__}")
print(f"Versao do NumPy: {numpy.__version__}")

caminho_imagem = r'C:\Users\gguim\Documents\MBA_Senac\Visao_Computacional\Foto.jpg'

# img_colorida = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
# if img_colorida is None:
#     raise IOError("file could not be read, check with os.path.exists()")
  
# img_original = cv2.imread(caminho_imagem, cv2.IMREAD_UNCHANGED)
# if img_original is None:
#     raise IOError("file could not be read, check with os.path.exists()")

img_cinza = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
if img_cinza is None:
    raise IOError("file could not be read, check with os.path.exists()")
  
edges = cv2.Canny(img_cinza, 100, 200)

cv2.imshow("Imagem cinza", img_cinza)
cv2.imshow("Canny edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()