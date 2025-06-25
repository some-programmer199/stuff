import pygame as p

# pygame setup
p.init()
screen = p.display.set_mode((566, 576))
clock = p.time.Clock()
running = True
dt = 0
friction= 100000
def convert(x,y):
    x=x*screen.get_width()/8
    y=y*screen.get_height()/8
    return x,y
player_pos = p.Vector2(convert(0,0))
background = p.image.load('board.png')


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in p.event.get():
        if event.type == p.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.blit(background,(0,0))

    p.draw.circle(screen, "red", player_pos, 10)

    keys = p.key.get_pressed()
    if keys[p.K_w]:
        player_pos.y += 500 * dt
    if keys[p.K_s]:
        player_pos.y-= 500 * dt
    if keys[p.K_a]:
        player_pos.x -= 500 * dt
    if keys[p.K_d]:
        player_pos.x += 500 * dt
    player_pos.y
    player_pos.x
    p.display.set_caption(str(player_pos))
    


    # flip() the display to put your work on screen
    p.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

p.quit()