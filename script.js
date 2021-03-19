
const SIZE = 1024
let vertex_data, region_data, tsne
let mode = 0

function setup() {
    createCanvas(SIZE, SIZE)
    d3.tsv("pca.tsv").then(function(data) {
        tsne = data.map(d => {
            return {
                x: parseFloat(d.x),
                y: parseFloat(d.y), 
                headline: d.headline
            }
        })
    })
    d3.tsv("vertices.tsv").then(function(data) {
        vertex_data = data
        d3.tsv("regions.tsv").then(function(data) {
            region_data = data.map(d => { 
                return {
                    elevation: parseFloat(d.elevation),
                    isEdge: d.is_edge == 'true',
                    coordinates: JSON.parse(d.coordinates),
                    headlines: d.headlines,
                    dominant_topic: parseInt(d.topics)
                }
            })
            console.log(region_data)
        })
    })
    drawForceGraph()
}

function drawTSNE() {

    document.getElementById('defaultCanvas0').hidden = false
    background('white')
    fill('black')
    xMax = d3.max(tsne.map(t => t.x))
    xMin = d3.min(tsne.map(t => t.x))
    yMax = d3.max(tsne.map(t => t.y))
    yMin = d3.min(tsne.map(t => t.y))
    let largest = d3.max([Math.abs(xMin), xMax, Math.abs(yMin), yMax])
    for(pt of tsne) {
        let x = (pt.x + largest) / largest / 2
        let y = (pt.y + largest) / largest / 2
        circle(x * SIZE, y * SIZE, 2)
    }
}

function reset() {
    document.getElementById('defaultCanvas0').hidden = true
    document.getElementById('regionsSVGContainer').hidden = true
}

function mouseClicked() {
    mode  = (mode + 1) % 3
    reset();
    if (mode == 0) {
        drawRegionsSVG()
    } else if (mode == 1) {
        drawTSNE()
    } else if (mode == 2) {
        drawForceGraph()
    }
}

function drawRegions() {
    document.getElementById('defaultCanvas0').hidden = false
    background('white')
    for(r of region_data) {
        noStroke()
        fill(230, 250, 255)
        if (r.elevation > 0) {
            fill(200 - r.elevation * 2)
        }
        beginShape()
        for(c of r.coordinates) {
            if (c != -1) {
                vertex(vertex_data[c].x * 1024, vertex_data[c].y * 1024)
            }
        }
        endShape(CLOSE)
    }
}

function drawRegionsSVG() {
    document.getElementById('regionsSVGContainer').hidden = false
    let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('viewBox', '0 0 1 1')
    svg.setAttribute('width', SIZE)
    svg.setAttribute('height', SIZE)
    var rainbow = d3.scaleSequential(d3.interpolateRainbow).domain([0, 100])
    for(const r of region_data) {
        if (r.isEdge) {
            continue
        }
        let blue = 'rgb(230, 250, 255)'
        let color = blue
        if (r.elevation > 0) {
            let hex = (200 - r.elevation * 2).toString(16)
            blue = `#${hex}${hex}${hex}`
            blue = rainbow(r.dominant_topic)
        }
        let poly = document.createElementNS('http://www.w3.org/2000/svg', 'path')
        let points = 'M'
        for(c of r.coordinates) {
            if (c != -1) {
                points += vertex_data[c].x + ',' + vertex_data[c].y + ' '
            }
        }
        points = points.slice(0, -1)
        poly.setAttribute('d', points)
        poly.setAttribute('style', `fill: ${blue}`)
        poly.addEventListener('mouseenter', () => {
            let tooltip = document.getElementById('tooltip')
            tooltip.innerHTML = `<p>${r.dominant_topic}</p>`
            let headlinesStr = r.headlines.slice(1, 1000)
            for(const st of headlinesStr.split('\', \'')) {
                tooltip.innerHTML += `<p>${st}</p>`
            }
            poly.classList.add('selected')
        })
        poly.addEventListener('mouseleave', () => {
            poly.classList.remove('selected')
        })
        svg.appendChild(poly)
    }
    console.log("done constructing")
    document.getElementById('svgContainer').innerHTML = ''
    document.getElementById('svgContainer').appendChild(svg)
}

function draw() {

}

function drawForceGraph() {

    let width = 1024;
    let height = 1024;
    let gcolor = function() {
        const scale = d3.scaleOrdinal(d3.schemeCategory10);
        return d => scale(parseInt(d.id));
      }();
    let graph = function(data) {
        console.log(data)
        const links = data.links.filter(d => parseFloat(d.value) > 0.1).map(d => {
            return { 
                source: parseInt(d.source),
                target: parseInt(d.target),
                value: parseFloat(d.value)
        }})
        const nodes = data.nodes.map(d => Object.create(d));
        let col = cola.d3adaptor(d3).size([width, height]);
    
        console.log(nodes)
        console.log(links)
        const simulation = col 
            .nodes(nodes)
            .links(links)
            .linkDistance(n => (1 - n.value ** 2) * 100)
            .start(30)
            //.force("link", d3.forceLink(links).id(d => d.id).distance(d => parseFloat(d.value) ** 4))
            //.force("charge", d3.forceManyBody().strength(-10))
            //.force("center", d3.forceCenter(width / 2, height / 2));
    
        const svg = d3.create("svg")
            .attr("viewBox", [0, 0, width, height]);
    
        const link = svg.append("g")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
        .selectAll("line")
        .data(links)
        .join("line")
            .attr("stroke-width", d => Math.sqrt(d.value));
    
        const node = svg.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
        .selectAll("circle")
        .data(nodes)
        .join("circle")
            .attr("r", 5)
            .attr("fill", gcolor)
            //.call(drag(simulation));
    
        node.append("title")
            .text(d => d.id);
    
        col.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
    
        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
        });
    
        //invalidation.then(() => simulation.stop());
    
        return svg.node();
    }
    d3.json("topic_adjacency.json").then((data) => {
        document.getElementById('graphContainer').appendChild(graph(data))
    })
}

