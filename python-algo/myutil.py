def div(a,b):
    if b != 0:
        value = float(a) / float(b)
    else:
        value = float('Inf')
    return value

def create_line(start, end):
    # inclusive
        
    dx = end[0]-start[0]
    dy = end[1]-start[1]

    if dx==0 and dy==0:
        return [start]
    elif dx == 0:
        extension,step = (1,1) if dy > 0 else (-1,-1)
        loc = list(map( lambda x: [start[0], start[1]+x], range(0,dy+extension,step) ))
        return loc
    elif dy == 0:
        extension,step = (1,1) if dx > 0 else (-1,-1)
        loc = list(map( lambda x: [start[0]+x, start[1]], range(0,dx+extension,step) ))
        return loc
    else:
        slope = float(dy)/float(dx)
        loc = []
        prev = start[1]
        extension,step = (1,1) if dx > 0 else (-1,-1)
        for i in range(0,dx+extension,step):
            x = start[0] + i
            y = int(start[1] + float(i)*slope)

            if prev is not None and abs(y-prev)>1:
                for j in range(abs(y-prev)):
                    sign = 1. if y-prev > 0 else -1.
                    yy = int(float(prev)+float(j)*sign)
                    # gamelib.debug_write('line: {}, {}, {}'.format(x,yy, sign))
                    loc.append([x,yy])
            else:
                # gamelib.debug_write('line: {}, {}'.format(x,y))
                loc.append([x,y])
            prev = y

        return loc
        
