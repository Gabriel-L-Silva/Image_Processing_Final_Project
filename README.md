# Eye tracking
Autores: Gabriel Lucas da Silva e Diogo Henrique Godoi

## Abstract
O objetivo deste projeto é desenvolver uma forma de detectar a região dos olhos. 

A entrada da aplicação será composta por 5 vídeos do conjunto de dados de clipes de 15s de vídeos do youtube adquiridos na base de dados [ChaLearn First impressions](https://chalearnlap.cvc.uab.cat/dataset/20/description/) . Segue alguns exemplos:


![image](https://user-images.githubusercontent.com/29693842/120404422-53f10a00-c314-11eb-852e-04eab7238944.png)
![image](https://user-images.githubusercontent.com/29693842/120404436-5ce1db80-c314-11eb-9e6b-205332b798f9.png)

A principal técnica de processamento de imagens utilizada será a morfologia. Nós utilizamos as técnicas do trabalho de [ "Rajpathak, Tanmay & Kumar, Ratnesh & Schwartz, Eric. (2009). Eye Detection Using Morphological and Color Image Processing".](https://www.researchgate.net/publication/237415739_Eye_Detection_Using_Morphological_and_Color_Image_Processing)


Para atingir o objetivo iremos seguir os seguintes passos:

  1. Remover fundo da imagem e possíveis ruídos. A técnica para acelerar esse processo foi pular alguns frames, e no final, a média de todos os frames processados é aplicada como uma máscara para todo o vídeo.
  
  2. Detectar a pele da pessoa e usar como máscara para removê-la da imagem.
  
  3. Detectar a posição do olho da pessoa. A técnica aqui foi baseada no trabalho de "Rajpathak, Tanmay & Kumar, Ratnesh & Schwartz, Eric. (2009)", eles consideram que sempre que o olho estiver própriamente iluminado haverá um ponto de reflexão nos olhos, e é justamente esse ponto de reflexão que dever ser explorado para a detecção dos olhos. 
  
  4. Realiza a detecção de olhos por meio da técnica Haar cascade. Para realizar a comparação com o nosso resultado.

  Todo o projeto foi realizado em conjunto durante chamada de voz e vídeo via Discord.
  
### Resultados
  Os resultados obtidos podem ser acessados [aqui](https://drive.google.com/drive/folders/12ZARRIYUNgqI2m7p3n1iCw50Ml1KOO57?usp=sharing).
### Discussão
  Nossa técnica tem dificuldade quando o sujeito para analise está usando óculos, ou quando a qualidade da imagem não é boa, por exemplo vídeo borrado que colocamos nos resultados. Em casos de sucesso, nossa técnica não é perfeita, porém tem resultado relativamente consistente quando as condições necessárias são atendidas, como boa iluminação, e boa qualidade da imagem.
### Conclusão
  Video de sucesso:
  
  [![Watch the video](https://github.com/Gls-Facom/Image_Processing_Final_Project/blob/main/videos/imgSucesso1.png)](https://drive.google.com/file/d/1FYnXcqsiNtxLn2VF7rBq8q-aVv-39xc5/view?usp=sharing)
  
  Video de falha:
  
    [![Watch the video](https://github.com/Gls-Facom/Image_Processing_Final_Project/blob/main/videos/imgFalha1.png)](https://drive.google.com/file/d/1OZarG-EaNgdGItou_Z2lOfMR4TJpaoqW/view?usp=sharing)  
  
  Video Haar Cascade:
  
  [![Watch the video](https://github.com/Gls-Facom/Image_Processing_Final_Project/blob/main/videos/imgHaar1.png)](https://drive.google.com/file/d/1oxl4FKvf88SptaDo6YeJPe3Nq1iDkePI/view?usp=sharing)  [![Watch the video](https://github.com/Gls-Facom/Image_Processing_Final_Project/blob/main/videos/imgHaar2.png)](https://drive.google.com/file/d/1OZarG-EaNgdGItou_Z2lOfMR4TJpaoqW/view?usp=sharing)
  



