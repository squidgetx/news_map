function setup() {
    let gdata;
    let scale = 32;
    let p_width = 200;
    let yMin = 0;
    let xMin = 0;
    let rainbow = d3.scaleSequential(d3.interpolateRainbow).domain([0, 20]);
    let getLeft = function(d) {
        return d.x * scale - xMin - p_width/2
    }
    let getTop = function(d) {
        return d.y * scale - yMin 
    }
    let animate = function(ele, index) {
        let text = ele.getAttribute('text')
        if (index == text.length) {
            window.setTimeout(function() {
                let nextHeadline = gdata[Math.floor(Math.random() * gdata.length)]
                ele.setAttribute('text', nextHeadline.headline)
                ele.style.left = getLeft(nextHeadline)
                ele.style.top = getTop(nextHeadline)
                ele.innerHTML = ''
                animate(ele, 0)
            }, 5000)
            return
        }
        //ele.innerHTML = text.slice(0, index)
        window.setTimeout(function() {animate(ele, index+1)}, 100)
    }
    d3.tsv("tsne.tsv").then(function(data) {
        // data is array with object
        // headline, x, y
        gdata = data.filter(d => d.x != '' )
        xMin = d3.min(data.map(d => d.x * scale))
        yMin = d3.min(data.map(d => d.y * scale))

        //data = data.filter(d => Math.random() < 0.01) 
        d3.select('#div')
            .data(data)
            .enter()
            .append('p')
            .style('left', d=> getLeft(d))
            .style('top', d=> getTop(d))
            .style('color', d => rainbow(d.t))
            .html('.')
            .attr('class', 'headline')
        let headlines = document.getElementsByClassName('headline')
        for(let i = 0; i < headlines.length; i++) {
            window.setTimeout(function() {
             //   animate(headlines[i], 0);
            }, i * 100)
        }
    })


}

