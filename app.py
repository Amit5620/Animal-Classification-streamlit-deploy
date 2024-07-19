import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

IMAGE_SIZE = (180, 180)
THRESHOLD = 0.5

st.set_page_config(
    page_title="Animal Classification",
    page_icon="👨‍⚕️"
)

st.header(':blue[Welcome to our Animal Classification website.] :syringe: :pill: :ambulance:', divider='rainbow')

placeholder = st.empty()

with placeholder.form("upload"):
    st.header(':yellow[Provide an Animal image.]:mag:', divider='orange')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Predict")

if submit and uploaded_file:
    placeholder.empty()

    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    # img_array = tf.expand_dims(img_array, 0)

    img = Image.open(uploaded_file)
    image = np.array(img)
    resized_image = cv2.resize(image, (180,180))
    resized_image = resized_image.reshape(-1,180,180,3)
    resized_image = np.array(resized_image)

    model = tf.keras.models.load_model('./models/animals.keras')
    predictions = model.predict(resized_image)
    max_index = predictions.argmax()

    labels = ["Bird", "Cat", "Dog"]
    label = labels[max_index]
    prediction = predictions[0][max_index]

    if prediction < THRESHOLD:
        label = "Unknown"
        prediction = 0.0

    prediction = round(prediction * 100, 2)

    st.header(f':mag: :rainbow[This is a : {label}]')
    st.write(f'Confidence {prediction}%.')

    st.image(img, caption='Uploaded Image')


    if label=='Bird':
        st.title('About Bird')
        st.write('''
                 Birds are vertebrate animals that have feathers, wings, and beaks. Like all vertebrates, they have bony skeleton. Most birds are able to fly, but some (like ostriches and penguins) cannot fly even though they still have wings. Other kinds of animals like insects and bats can fly too, but birds are the only animals with feathers. Birds have scales on their lower legs and feet, like reptiles. All birds reproduce by laying eggs with hard shells, and most build nests to help protect the eggs from weather and predators. Adult birds almost always sit on the eggs to keep them warm until they hatch.

                Like mammals, birds are warm-blooded which means that they make their own body heat and can stay warm even when the sun is not out. Birds mostly eat high-energy foods like seeds and fruits, insects and other animals, nectar, or meat from dead animals. Very few birds eat other plant parts, like leaves or roots, which are more difficult to digest. Many birds feed insects to their babies to help them grow fast even if the adult birds eat seeds or fruits. The shape and size of bird beaks are adapted to the kinds of food they eat.

                Birds can be found in all habitats above ground and there are even some species that make nests in underground burrows. Most birds try to lay their eggs in places that predators will not be able to reach easily.
                 ''')
        
    elif label=='Cat':
        st.title('About Cat')
        st.write('''
                 The cat (Felis catus), commonly referred to as the domestic cat or house cat, is a small domesticated carnivorous mammal. It is the only domesticated species of the family Felidae. Recent advances in archaeology and genetics have shown that the domestication of the cat occurred in the Near East around 7500 BC. It is commonly kept as a house pet and farm cat, but also ranges freely as a feral cat avoiding human contact. Valued by humans for companionship and its ability to kill vermin, the cat's retractable claws are adapted to killing small prey like mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth, and its night vision and sense of smell are well developed. It is a social species, but a solitary hunter and a crepuscular predator. Cat communication includes vocalizations like meowing, purring, trilling, hissing, growling, and grunting as well as cat body language. It can hear sounds too faint or too high in frequency for human ears, such as those made by small mammals. It secretes and perceives pheromones.
                 ''')
        
    elif label=='Dog':
        st.title('About Dog')
        st.write('''
                 The dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf. Also called the domestic dog, it was domesticated from an extinct population of Pleistocene wolves over 14,000 years ago. The dog was the first species to be domesticated by humans. Experts estimate that hunter-gatherers domesticated dogs more than 15,000 years ago, which was before the development of agriculture. Due to their long association with humans, dogs have expanded to a large number of domestic individuals and gained the ability to thrive on a starch-rich diet that would be inadequate for other canids.[4]

                The dog has been selectively bred over millennia for various behaviors, sensory capabilities, and physical attributes.[5] Dog breeds vary widely in shape, size, and color. They perform many roles for humans, such as hunting, herding, pulling loads, protection, assisting police and the military, companionship, therapy, and aiding disabled people. Over the millennia, dogs became uniquely adapted to human behavior, and the human–canine bond has been a topic of frequent study. This influence on human society has given them the sobriquet of "man's best friend".
                 ''')
        


