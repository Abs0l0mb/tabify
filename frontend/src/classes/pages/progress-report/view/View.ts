import {
    Block,
    Div,
    Chart,
    Timeline
} from '@src/classes';

export interface ViewOptions {
    label: string
    data: Record<number, any>
    editable: boolean
    maxTime: number
}

export type ViewUpdateCallback = (updatedCategoryTimes: Record<number, number>) => void;

export class View extends Div {

    private header: Div;
    private label: Div;
    private time: Div;

    private content: Div;
    public chart: Chart;
    public timeline: Timeline;

    private categoryTimes: Record<number, number> = {};

    constructor(options: ViewOptions, parent: Block) {

        super('view', parent);

        for (const categoryId in options.data)
            this.categoryTimes[Number(categoryId)] = options.data[Number(categoryId)].totalTime;

        //======
        //HEADER
        //======

        this.header = new Div('header', this);

        this.label = new Div('label', this.header).write(options.label);
        this.time = new Div('time', this.header).write("Total Work Time: 0h");

        //=======
        //CONTENT
        //=======

        this.content = new Div('content', this);

        //=====
        //CHART
        //=====

        this.chart = new Chart(options.data, {
            editable: options.editable
        }, this.content);
        
        this.chart.on('update', (updatedTimes: Record<number, number>) => {

            this.categoryTimes = updatedTimes;
            
            for (const catId in updatedTimes)
                this.timeline.updateCategoryTime(parseInt(catId, 10), updatedTimes[catId]);
            
            this.updateTime();
            
            this.emit('update', this.categoryTimes);
        });

        //========
        //TIMELINE
        //========

        this.timeline = new Timeline(options.data, this.content, {
            editable: options.editable,
            maxTime: options.maxTime
        });
        
        this.timeline.on('update', (data: any) => {

            this.categoryTimes[data.categoryId] = data.newTime;
            
            this.chart.updateCategoryTime(data.categoryId, data.newTime);
            this.updateTime();
            
            this.emit('update', this.categoryTimes);
        });

        this.updateTime();
    }

    /*
    **
    **
    */
    private updateTime() : void {

        let totalTime = 0;

        for (const key in this.categoryTimes)
            totalTime += this.categoryTimes[key];
        
        this.time.write(`Total Work Time: ${totalTime}h`);
    }

    /*
    **
    **
    */
    public updateCategoryTime(categoryId: number, newTime: number) : void {

        this.categoryTimes[categoryId] = newTime;
        this.chart.updateCategoryTime(categoryId, newTime);
        this.timeline.updateCategoryTime(categoryId, newTime);

        this.updateTime();
    }
}
