import halcon as ha
import os
import pathlib

pathlib.Path(__file__).parent.resolve()


if __name__ == "__main__":
    main_path = os.path.dirname(os.path.abspath(__file__))
    preprocessed_path = main_path + "/preprocessed/"
    imgs_names = sorted(os.listdir(preprocessed_path))
    example = preprocessed_path + imgs_names[0]

    ocr_model = ha.create_text_model_reader("auto", "Industrial_NoRej.omc")
    image = ha.read_image(example)
    prediction = ha.find_text(image, ocr_model)

    obj_results = ha.get_text_object(prediction, "all_lines")
    char_results = ha.get_text_result(prediction, "class")
    acc_results = ha.get_text_result(prediction, "confidence")

    characters = []
    for i in range(len(obj_results)):
        areas, rows, cols = ha.area_center(obj_results[i])
        character = char_results[i]
        position = tuple([rows[0], cols[0]])
        confidence = acc_results[i]

        characters.append(tuple([character, position, confidence]))

    # Filter results with a confidence lower than ...
    characters = filter(lambda el: el[-1] > 0.7, characters)
    for r in characters:
        print(r)
