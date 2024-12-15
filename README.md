Здесь располагается Ai проект "Fingers", который собирает данные жеста руки и распознает его.
Скрипт для использования делает распознование в реальном времени и рисует букву на руке.
Создание модели проиводится во фреймворке Keras (tensorflow) и обучения ее на созданном дадасете.

Состав файлов:
- fingers2 - colab файл для парсинга данных и обучения модели
- dataset.zip - самописный датасет для обучения модели
- modelTest.py - проверка совместимости на лькальной машине (сохраненная модель от разных версий keras может конфликтовать). Если файл говорит - ошибка, выравнивайте версии tensorflow

  

Заголовки для DataFrame с названиями точек одной руки Mediapipe
 
titles = [

    'wrist_x', 'wrist_y', 'wrist_z',          # Запястье    
    'thumb_cmc_x', 'thumb_cmc_y', 'thumb_cmc_z',  # Основание большого пальца
    'thumb_mcp_x', 'thumb_mcp_y', 'thumb_mcp_z',  # Сустав большого пальца
    'thumb_ip_x', 'thumb_ip_y', 'thumb_ip_z',      # Кончик большого пальца
    'thumb_tip_x', 'thumb_tip_y', 'thumb_tip_z',   # Кончик большого пальца
    

    'index_finger_mcp_x', 'index_finger_mcp_y', 'index_finger_mcp_z',  # Сустав указательного пальца
    'index_finger_pip_x', 'index_finger_pip_y', 'index_finger_pip_z',  # Средний сустав указательного пальца
    'index_finger_dip_x', 'index_finger_dip_y', 'index_finger_dip_z',  # Конечный сустав указательного пальца
    'index_finger_tip_x', 'index_finger_tip_y', 'index_finger_tip_z',  # Кончик указательного пальца
    

    'middle_finger_mcp_x', 'middle_finger_mcp_y', 'middle_finger_mcp_z',  # Сустав среднего пальца
    'middle_finger_pip_x', 'middle_finger_pip_y', 'middle_finger_pip_z',  # Средний сустав среднего пальца
    'middle_finger_dip_x', 'middle_finger_dip_y', 'middle_finger_dip_z',  # Конечный сустав среднего пальца
    'middle_finger_tip_x', 'middle_finger_tip_y', 'middle_finger_tip_z',  # Кончик среднего пальца

    'ring_finger_mcp_x', 'ring_finger_mcp_y', 'ring_finger_mcp_z',  # Сустав безымянного пальца
    'ring_finger_pip_x', 'ring_finger_pip_y', 'ring_finger_pip_z',  # Средний сустав безымянного пальца
    'ring_finger_dip_x', 'ring_finger_dip_y', 'ring_finger_dip_z',  # Конечный сустав безымянного пальца
    'ring_finger_tip_x', 'ring_finger_tip_y', 'ring_finger_tip_z',  # Кончик безымянного пальца

    'pinky_mcp_x', 'pinky_mcp_y', 'pinky_mcp_z',  # Сустав мизинца
    'pinky_pip_x', 'pinky_pip_y', 'pinky_pip_z',  # Средний сустав мизинца
    'pinky_dip_x', 'pinky_dip_y', 'pinky_dip_z',  # Конечный сустав мизинца
    'pinky_tip_x', 'pinky_tip_y', 'pinky_tip_z',   # Кончик мизинца

    'class' # класс
]


#surdo #ai #ml #LEbEdEV
