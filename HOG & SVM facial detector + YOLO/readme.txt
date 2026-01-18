1. Bibliotecile necesare pentru rularea proiectului, inclusiv versiunea completa a fiecarei biblioteci.

numpy==2.2.3
opencv-python==4.11.0
matplotlib==3.10.8
scikit-learn==1.8.0
torch==2.7.1+cu118
ultralytics==8.3.248

Versiunea de Python utilizata: 3.10.9

2. Modul de rulare a codului si unde pot fi gasite fisierele de output.

Pentru a rula cu folderul de testare:
    trebuie modificata linia 8 din folderul Bonus din generare_rezultate_finale.py
    trebuie modificate liniile 9 si 10 din Parameters.py din Task 1
    trebuie modificate liniile 11-16 din Parameters.py din Task 2


script: RunProject.py
Cum se ruleaza: 
- Navigati in directorul task-ului pe care doriti sa il rulati.
- Executati comanda: python RunProject.py
- Pentru task-ul bonus, navigati in directorul "Bonus" si rulati: python generare_rezultate_finale.py

output: 
- Toate rezultatele .npy atat pentru task-urile normale cat si pentru bonus sunt salvate in folderul principal al proiectului, in directorul "fisiere_solutie".
- Nota: Daca doriti sa rulati de la zero (recalculare completa), stergeti continutul folderului "data/salveazaFisiere".