from dataset.scripts.frame_acquisition import frame_acquisition

frames_to_acquire = [
    'images/prova2.jpg',
    'images/prova1.jpg'
]

for fr in frames_to_acquire:
    frame_acquisition(fr)
