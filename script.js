const SIZE = 1024
let vertex_data, region_data, tsne, topics
let topicSizes = topicMediaNames = []
let mode = 0
let date = "2020-02-01"
let end_date = ''
let interval = 28
const DAY_MS = 1000 * 60 * 60 * 24
const TOPIC = 'us_mainstream_stories_trunc'

function getName(date, interval, suffix){
    let start = new Date(date)
    start.setUTCHours(start.getUTCHours() + 4);
    let end = new Date(start)
    end.setDate(start.getDate() + interval)
    let startStr = start.toISOString().slice(0, 10)
    let endStr = end.toISOString().slice(0, 10)
    end_date = endStr 
    let str = `data/${TOPIC}_${startStr}_${endStr}.${suffix}`
    console.log(str)
    return str

}

function setup() {
    //createCanvas(SIZE, SIZE)
   drawMaster()
    
}

function drawMaster() {

let topicName = getName(date, interval, 'topics.json')
    d3.json(topicName).then((topicData) => {
        topics = topicData
        topicSizes = []
        for(k in topics) {
            topicSizes[k] = topics[k]["_metadata_"]["total"]
            topicMediaNames[k] = topics[k]["_metadata_"]["media_names"]
            delete topics[k]["_metadata_"]
        }
        let min = '2020-02-01';
        let max = '2020-02-22';
        let prevDisabled = ''
        let nextDisabled = ''
        if (date == min) {
            prevDisabled = 'disabled'
        }
        if (date == max) {
            nextDisabled = 'disabled'
        }
        //document.getElementById('graphContainer').innerHTML = `<h2>News Landscape ${monthStr} 2020</h2>`
        document.getElementById('graphContainer').innerHTML = `<h2>2020 News Landscape </h2>`
        document.getElementById('graphContainer').innerHTML += 
            `<button onclick="prevMonth()" ${prevDisabled}>prev</button> `
        document.getElementById('graphContainer').innerHTML += `<span class="month">${date}-${end_date}</span> `
        document.getElementById('graphContainer').innerHTML += 
            `<button onclick="nextMonth()" ${nextDisabled}>next</button>`
        drawLayoutGraph()
        drawRegionsSVG()
    })
}

function nextMonth() {
    let d = new Date(date)
    d.setUTCHours(d.getUTCHours() + 4);
    let nextDate = new Date(d)
    nextDate.setDate(d.getDate() + 7);
    date = nextDate.toISOString().slice(0, 10)
    drawMaster()
}
function prevMonth() {
    let d = new Date(date)
    d.setUTCHours(d.getUTCHours() + 4);
    let nextDate = new Date(d)
    nextDate.setDate(d.getDate() - 7);
    date = nextDate.toISOString().slice(0, 10)
    drawMaster()
}

function keyPressed() {
    if (keyCode === LEFT_ARROW) {
        month -= 1
    } else if (keyCode === RIGHT_ARROW) {
        month += 1
    }
    drawMaster()
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
    for(const pt of tsne) {
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
    /*
    mode  = (mode + 1) % 3
    reset();
    if (mode == 0) {
        drawRegionsSVG()
    } else if (mode == 1) {
        drawTSNE()
    } else if (mode == 2) {
        drawForceGraph()
    }
    */
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
    let renderRegionsSVG = function(region_data, vertex_data) {
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
            if (r.elevation > 0) {
                let hex = (Math.floor(map(r.elevation, 0, 5000, 150, 0))).toString(16)
                blue = `#${hex}${hex}${hex}`
                //blue = rainbow(r.dominant_topic)
            }
            let poly = document.createElementNS('http://www.w3.org/2000/svg', 'path')
            let points = 'M'
            for(const c of r.coordinates) {
                if (c != -1) {
                    points += vertex_data[c].x + ',' + vertex_data[c].y + ' '
                }
            }
            points = points.slice(0, -1)
            poly.setAttribute('d', points)
            poly.setAttribute('style', `fill: ${blue}`)
            poly.addEventListener('mouseenter', () => {
                let tooltip = document.getElementById('tooltip')
                tooltip.innerHTML = `<p>${r.elevation}</p>`
                //let headlinesStr = r.headlines.slice(1, 1000)
                //for(const st of headlinesStr.split('\', \'')) {
                //    tooltip.innerHTML += `<p>${st}</p>`
               // }
                poly.classList.add('selected')
            })
            poly.addEventListener('mouseleave', () => {
                poly.classList.remove('selected')
            })
            svg.appendChild(poly)
        }
        return svg
    }
    d3.tsv(getName(date, interval, 'vertices.tsv')).then(vertexData => {
        console.log(getName(date, interval, 'regions.tsv'))
        vertexData = vertexData.map((d, i) => { return {
            id: i,
            x: parseFloat(d.x),
            y: parseFloat(d.y),
        }})
        d3.tsv(getName(date, interval, 'regions.tsv')).then(regionData => {
            regionData = regionData.map((d, i) => { return {
                id: i,
                coordinates: JSON.parse(d.coordinates),
                elevation: parseFloat(d.elevation),
                headlines: [d.elevation],
                is_edge: d.is_edge === 'True',
                dominant_topic: JSON.parse(d.topics)[0],
            }})
            console.log(vertexData)
            console.log(regionData)
            let svg = renderRegionsSVG(regionData, vertexData)
            console.log("done constructing")
            document.getElementById('svgContainer').innerHTML = ''
            document.getElementById('svgContainer').appendChild(svg)

        })
    })
}

function draw() {

}

let handleMouseOver = function(ev, d) {
    d3.select(this).transition()
        .duration(0.1)
        .attr("r", d => Math.sqrt(topicSizes[int(d.id)]) * 0.045)
    let sorted = Object.entries(topics[d.id]).filter(a => a[1] > 2).sort(function(a, b) {
        return b[1] - a[1];
    });
    let nodes = sorted.map(d => `<p class='word' style='font-size:${Math.sqrt(d[1]) + 6}pt'>${d[0]}</p>`).slice(0, 20).join("")
    document.getElementById('wordContainer').innerHTML = `<p>${d.id}</p>`
    let medianodes = ''
    for (mname in topicMediaNames[d.id]) {
        if (topicMediaNames[d.id][mname] > 2) {
            medianodes += `<p class='sm'>${mname}: ${topicMediaNames[d.id][mname]}`
        }
    }
    document.getElementById('wordContainer').innerHTML = d.id
    document.getElementById('wordContainer').innerHTML += '<h3>Commonly spoken words in this cluster</h3>'
    document.getElementById('wordContainer').innerHTML += nodes
    document.getElementById('wordContainer').innerHTML += '<h3>Breakdown of headlines in this cluster by media outlet'
    document.getElementById('wordContainer').innerHTML += medianodes
}

let handleMouseOut = function(ev, d) {
 d3.select(this).transition()
        .duration(0.1)
        .attr("r", d => Math.sqrt(topicSizes[int(d.id)]) * 0.04)
}

function drawLDAVis() {
    let width = 2;
    let height = 2;
    let topics = [];
    
   

    let vis = function(topic_points) {
        // array of id, x, y, and eventually sizes
        let gcolor = function() {
            const scale = d3.scaleOrdinal(d3.schemeCategory10);
            return d => scale(parseInt(d.id));
        }()
        const svg = d3.create("svg")
            .attr("viewBox", [-width/2, -height/2, width, height]);

        const nodes = svg.append("g")
            .selectAll("circle")
            .data(topic_points)
            .join("g")

        nodes
            .append('circle')
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => Math.sqrt(d.total) * 0.001)
            .attr("fill", gcolor)
            .on('mouseover', handleMouseOver)
            .on('mouseout', handleMouseOut)
        nodes
            .append('text')
            .style("font-size", d => Math.sqrt(d.total) * 0.001)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .html(d => d.id)
        return svg.node()
    }

    let tsvname = getName(date, interval, 'mds.tsv')
    d3.tsv(tsvname).then((data) => {
        distance_data = data.map(d => {
            return {
                x: parseFloat(d[0]),
                y: parseFloat(d[1]),
                id: parseInt(d[''])
            }
        })
        console.log('loading topics')
        d3.json(`topics_${TOPIC}_2020-${monthStr}-01_2020-${monthStr2}-01.json`).then((data) => {
            topics = data
            for(k in distance_data) {
                distance_data[k]["total"] = topics[k]["total"]
                delete topics[k]['total']
            }
            let node = vis(distance_data)
            document.getElementById('graphContainer').innerHTML = `<h2>${month}</h2>`
            document.getElementById('graphContainer').appendChild(node)
        })
    })
}

function drawForceGraph() {
    let width = 1024;
    let height = 1024;
    let gcolor = function() {
        const scale = d3.scaleOrdinal(d3.schemeCategory10);
        return d => scale(parseInt(d.id));
      }();
    let graph = function(data) {
        const links = data.links.map(d => {
            return { 
                source: parseInt(d.source),
                target: parseInt(d.target),
                value: parseFloat(d.value)
        }})
        const nodes = data.nodes.map(d => Object.create(d));
    
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d=>d.id))
            //.force("collision", d => Math.sqrt(topicSizes[int(d.id)] * 0.3))
            .force("charge", d3.forceManyBody().strength(d => -Math.sqrt(topicSizes[int(d.id)]) * 1.2))
            .force("center", d3.forceCenter(width / 2, height / 2).strength(1))
            
    
        const svg = d3.create("svg")
            .attr("viewBox", [0, 0, width, height]);
    
        const link = svg.append("g")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
        .selectAll("line")
        .data(links)

        const linklines = link.join("line")
            .attr("stroke-width", d => Math.sqrt(d.value));
        const linklabels = link.join("text")
            .attr('class' ,'linklabel')
            .html(d=> d.value.toFixed(2))
    
        const node = svg.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
        .selectAll("circle")
        .data(nodes)
        .join("circle")
            .attr("r", d => Math.sqrt(topicSizes[int(d.id)]) * 0.3)
            .attr("fill", gcolor)
            .on('mouseover', handleMouseOver)
            .on('mouseout', handleMouseOut)
    
        node.append("title")
            .text(d => d.id);
    
        simulation.on("tick", () => {
            linklines
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            linklabels
                .attr('x', d=> (d.source.x + d.target.x )/ 2)
                .attr('y', d=> (d.source.y + d.target.y) / 2)
        
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            });
    
        //invalidation.then(() => simulation.stop());
    
        return svg.node();
    }
    let jsonName = getName(date, interval, 'topic_adjacency.json')
    console.log(jsonName)
    d3.json(jsonName).then((data) => {
        document.getElementById('graphContainer').appendChild(graph(data))
    })
}


function drawLayoutGraph() {
    let width = 1024;
    let height = 1024;
    let gcolor = function() {
        const scale = d3.scaleOrdinal(d3.schemeCategory10);
        return d => scale(parseInt(d.group));
      }();
    let graph = function(data) {
        const nodes = data.layouts.map((layout, i) => layout.map(d => {
            return { 
                group: i,
                id: parseInt(d.id),
                x: parseFloat(d.x),
                y: parseFloat(d.y)
        }})).flat()
        let xExt = d3.extent(nodes.map(n => n.x))
        let yExt = d3.extent(nodes.map(n => n.y))
        console.log(nodes)

        const svg = d3.create("svg")
            .attr("viewBox", [xExt[0], yExt[0], xExt[1], yExt[1]]);
    
        const node = svg.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 0.1)
        .selectAll("circle")
        .data(nodes)
        .join("circle")
            .attr("r", d => Math.sqrt(topicSizes[int(d.id)]) * 0.04)
            .attr("cx", d=> d.x )
            .attr("cy", d=> d.y )
            .attr("fill", gcolor)
            .on('mouseover', handleMouseOver)
            .on('mouseout', handleMouseOut)
    
        return svg.node();
    }
    let jsonName = getName(date, interval, 'layout.json')
    console.log(jsonName)
    d3.json(jsonName).then((data) => {
        document.getElementById('graphContainer').appendChild(graph(data))
    })
}

