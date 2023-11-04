import pygame

display_size = (920, 720)
display_color = "black"
vis_rect = (0, 0, 620, 620)
vis_color = (30, 30, 30)
vis_center = (vis_rect[2] / 2, vis_rect[3] / 2)

road_radius = 300
road_width = 40
road_color = (255, 255, 255)

car_image = pygame.image.load("assets/car_1.png")
truck_image = pygame.image.load("assets/car-truck5.png")

car_image_rect = car_image.get_rect()


ui_rect = (0, 620, 920, 100)
start_stop_rect = (620, 0, 100, 30)

playback_rect = (620 + 10, 50)
iteration_rect = (620 + 10, 50 + 20)
outcome_rect = (620 + 10, 50 + 20 + 20)

progress_rect = (0, 620 + 10, 920, 30)
