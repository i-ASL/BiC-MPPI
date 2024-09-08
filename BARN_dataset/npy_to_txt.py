import numpy as np
import matplotlib.pyplot as plt

# for i in range(300):
for i in range(299,-1,-1):
    npy_file = f'grid_files/grid_{i}.npy'
    txt_file = f'txt_files/output_{i}.txt'

    raw_data = np.load(npy_file)
    raw_data = np.array(raw_data, dtype=np.float64)
    bracket = [1] + [0 for _ in range(28)] + [1]
    bracket = np.array(bracket, dtype=np.float64)
    for _ in range(10):
        raw_data = np.insert(raw_data, 0, bracket, axis = 1)
    for _ in range(10):
        raw_data = np.insert(raw_data, raw_data.shape[1], bracket, axis = 1)

    # plt.imshow(raw_data, cmap = 'Greys')

    rows = len(raw_data)
    cols = len(raw_data[0])

    BLOCK = 10
    raw_data[raw_data==1] = BLOCK
                        
    data = np.array(raw_data)
    for row in range(rows):
        for col in range(cols):
            if raw_data[row][col] == BLOCK:
                for nr in [row + 1, row - 1]:
                    if 0 <= nr < rows:
                        data[nr][col] = BLOCK    
                for nc in [col + 1, col - 1]:
                    if 0 <= nc < cols:
                        data[row][nc] = BLOCK
    raw_data[raw_data==1] = BLOCK
    
    np.savetxt(txt_file, data, fmt='%.5e')

