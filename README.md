# Eye tracking
Autores: Gabriel Lucas da Silva e Diogo Henrique Godoi

## Abstract
O objetivo deste projeto é desenvolver uma forma de detectar a região dos olhos. 

A entrada da aplicação será composta por um conjunto de dados de clipes de 15s de vídeos no youtube adquiridos na base de dados https://chalearnlap.cvc.uab.cat/dataset/20/description/ . Segue alguns exemplos:


![image](https://user-images.githubusercontent.com/29693842/120404422-53f10a00-c314-11eb-852e-04eab7238944.png)
![image](https://user-images.githubusercontent.com/29693842/120404436-5ce1db80-c314-11eb-9e6b-205332b798f9.png)

A principal técnica de processamento de imagens utilizada será a morfologia. Nós utilizamos as técnicas do trabalho de [ "Rajpathak, Tanmay & Kumar, Ratnesh & Schwartz, Eric. (2009). Eye Detection Using Morphological and Color Image Processing".](https://www.researchgate.net/publication/237415739_Eye_Detection_Using_Morphological_and_Color_Image_Processing)


Para atingir o objetivo iremos seguir os seguintes passos:

  1. Remover fundo da imagem e possíveis ruídos
  
  
  
  2. Detectar a face da pessoa
  
  3. Detectar a posição do olho da pessoa
      
      A técnica aqui foi baseada no trabalho de "Rajpathak, Tanmay & Kumar, Ratnesh & Schwartz, Eric. (2009)", eles consideram que sempre que o olho estiver própriamente iluminado haverá um ponto de reflexão nos olhos, e é justamente esse ponto de reflexão que dever ser explorado para a detectção dos olhos.
    
  


### Resultados
  Os resultados obtidos podem ser acessados [aqui](https://drive.google.com/drive/folders/12ZARRIYUNgqI2m7p3n1iCw50Ml1KOO57?usp=sharing).
### Discussão

### Conclusão
  


