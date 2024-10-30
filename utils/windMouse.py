import numpy

sqrt3 = numpy.sqrt(3)
sqrt5 = numpy.sqrt(5)


def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=4, W_0=1.5, M_0=20, D_0=8, move_mouse=lambda x, y: None):
    """
    Оптимизированный WindMouse алгоритм для плавного и быстрого перемещения мыши.
    G_0 - сила притяжения к цели (уменьшена для увеличения скорости).
    W_0 - сила ветра (уменьшена для снижения случайных отклонений).
    M_0 - максимальный шаг (уменьшен для увеличения контролируемости движений).
    D_0 - расстояние, при котором поведение ветра меняется с случайного на затухающие.
    """
    current_x, current_y = start_x, start_y
    v_x = v_y = W_x = W_y = 0

    sqrt3 = 1.732
    sqrt5 = 2.236

    while (dist := numpy.hypot(dest_x - start_x, dest_y - start_y)) >= 2:  # Останавливаемся при расстоянии меньше 2
        W_mag = min(W_0, dist)  # Уменьшение силы ветра при приближении
        if dist >= D_0:
            W_x = W_x / sqrt3 + (2 * numpy.random.random() - 1) * W_mag / sqrt5
            W_y = W_y / sqrt3 + (2 * numpy.random.random() - 1) * W_mag / sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = numpy.random.random() * 3 + 3
            else:
                M_0 /= sqrt5

        # Увеличение силы притяжения для ускорения движения
        v_x += W_x + G_0 * (dest_x - start_x) / dist
        v_y += W_y + G_0 * (dest_y - start_y) / dist

        # Ограничение по максимальному шагу для плавности движения
        v_mag = numpy.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0 / 2 + numpy.random.random() * M_0 / 2
            v_x = (v_x / v_mag) * v_clip
            v_y = (v_y / v_mag) * v_clip

        # Обновление позиции
        start_x += v_x
        start_y += v_y
        move_x = int(numpy.round(start_x))
        move_y = int(numpy.round(start_y))

        # Движение мыши только при необходимости
        if current_x != move_x or current_y != move_y:
            move_mouse(current_x := move_x, current_y := move_y)

    # Возвращаем последнюю позицию
    return current_x, current_y
