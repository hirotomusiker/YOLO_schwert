
def add_neighbours(xi, x, yi, y, ai, ti, fsize, d=0.5):
    """ Experimental function to assign target grids
    """
    x = torch.cat([x - xi, x - xi + d, x - xi - d, x - xi, x - xi])
    y = torch.cat([y - yi, y - yi, y - yi, y - yi + d, y - yi - d])
    xi = torch.cat([xi, xi - 1, xi + 1, xi, xi])
    yi = torch.cat([yi, yi, yi, yi - 1, yi + 1])
    ai = torch.cat([ai, ai, ai, ai, ai])
    ti = torch.cat([ti, ti, ti, ti, ti])
    ind = (0 <= xi) & (xi < fsize) & (0 <= yi) & (yi < fsize) & (0 < x) & (0 < y) & (x < 1) & (y < 1)
    xi = xi[ind]
    yi = yi[ind]
    ai = ai[ind]
    ti = ti[ind]
    return xi, yi, ai, ti


def add_neighbours_3x3(xi, x, yi, y, ai, ti, fsize, d=0.5):
    """ Experimental function to assign target grids
    """
    x = torch.cat([x - xi, x - xi + d, x - xi - d, x - xi, x - xi, x - xi + d, x - xi - d, x - xi + d, x - xi - d])
    y = torch.cat([y - yi, y - yi, y - yi, y - yi + d, y - yi - d, y - yi + d, y - yi + d, y - yi - d, y - yi - d])
    xi = torch.cat([xi, xi - 1, xi + 1, xi, xi, xi - 1, xi + 1, xi - 1, xi + 1])
    yi = torch.cat([yi, yi, yi, yi - 1, yi + 1, yi - 1, yi - 1, yi + 1, yi + 1])
    ai = torch.cat([ai, ai, ai, ai, ai, ai, ai, ai, ai])
    ti = torch.cat([ti, ti, ti, ti, ti, ti, ti, ti, ti])
    ind = (0 <= xi) & (xi < fsize) & (0 <= yi) & (yi < fsize) & (0 < x) & (0 < y) & (x < 1) & (y < 1)
    xi = xi[ind]
    yi = yi[ind]
    ai = ai[ind]
    ti = ti[ind]
    return xi, yi, ai, ti
