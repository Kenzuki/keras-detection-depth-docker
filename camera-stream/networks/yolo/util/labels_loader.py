class LabelsLoader:
    __labels_en = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                   "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                   "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                   "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                   "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                   "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                   "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                   "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                   "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                   "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    __labels_pl = ["osoba", "rower", "samochód", "motocykl", "samolot", "autobus", "pociąg", "ciężarówka",
                   "łódź", "sygnalizacja świetlna", "hydrant", "znak stopu", "parkometr", "ławka",
                   "ptak", "kot", "pies", "koń", "owca", "krowa", "słoń", "niedźwiedź", "zebra", "żyrafa",
                   "plecak", "parasol", "torebka", "krawat", "walizka", "frisbee", "narty", "snowboard",
                   "piłka sportowa", "latawiec", "kij baseballowy", "rękawica baseballowa", "deskorolka",
                   "deska surfingowa",
                   "rakieta tenisowa", "butelka", "kieliszek do wina", "kubek", "widelec", "nóż", "łyżka", "miska",
                   "banan",
                   "jabłko", "kanapka", "pomarańcza", "brokuł", "marchew", "hot dog", "pizza", "pączek", "ciasto",
                   "krzesło", "kanapa", "roślina doniczkowa", "łóżko", "stół obiadowy", "toaleta", "telewizor",
                   "laptop", "myszka",
                   "pilot", "klawiatura", "telefon komórkowy", "mikrofalówka", "piekarnik", "toster", "umywalka",
                   "lodówka",
                   "książka", "zegar", "waza", "nożyczki", "pluszowy miś", "suszarka do włosów", "szczoteczka do zębów"]

    @staticmethod
    def load(lang: str):
        if lang is None or lang == "en":
            return LabelsLoader.__labels_en
        elif lang == "pl":
            return LabelsLoader.__labels_pl
        else:
            raise Exception("No labels for language: %s" % lang)
