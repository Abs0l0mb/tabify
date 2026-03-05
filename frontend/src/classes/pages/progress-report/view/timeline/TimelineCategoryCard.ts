
import { 
    Div ,
    Timeline,
    TimelineOptions,
    Block,
    ProgressReport
} from '@src/classes';

export class TimelineCategoryCard extends Div {
    
    public currentTotalTime: number;
    private totalTimeDiv: Div;
    
    private documentBlock: Block = new Block(document);
    
    constructor(
        private category: any,
        private options: TimelineOptions,
        private timeline: Timeline
    ) {
        
        super('timeline-category-card', timeline);
        
        //=====
        //TITLE
        //=====
        
        const titleDiv = new Div('category-title', this).write(category.category_title);
        
        //=========
        //TASK LIST
        //=========

        const tasksListDiv = new Div('tasks-list', this);
        
        let totalTime = 0;
        
        category.tasks.forEach((task: any) => {

            new Div('task', tasksListDiv).write(`${task.task_title}`); 
            
            totalTime += task.duree;
        });
        
        //==========
        //TOTAL TIME
        //==========
        
        this.currentTotalTime = totalTime;

        this.totalTimeDiv = new Div('total-time', this).write(this.currentTotalTime.toString());
        
        this.setStyle('width', `${(this.currentTotalTime * 100) / this.options.maxTime}%`);
        
        //=======
        //EDITION
        //=======

        if (this.options.editable) {
            
            const resizer = new Div('resizer', this);
            
            resizer.onNative('mousedown', (e: MouseEvent) => {

                e.preventDefault();
                const startX = e.clientX;
                const startWidth = this.element.getBoundingClientRect().width;
                const container = this.element.parentElement;
                const containerWidth = container ? container.getBoundingClientRect().width : 1;
                
                const onMouseMove = (e: MouseEvent) => {

                    const dx = e.clientX - startX;
                    let newWidth = startWidth + dx;
                    const minWidth = 20;
                    if (newWidth < minWidth) newWidth = minWidth;
                    let newWidthPercentage = (newWidth / containerWidth) * 100;
                    
                    let newTotalTime = Math.round((newWidthPercentage * this.options.maxTime) / 100);
                    
                    let sumOther = 0;
                    
                    for (const key in this.timeline.timelineCards) {
                        
                        const card = this.timeline.timelineCards[key];
                        if (card !== this) {
                            sumOther += card.currentTotalTime;
                        }
                    }
                    
                    const maxAvailable = this.options.maxTime - sumOther;
                    
                    if (newTotalTime > maxAvailable) {
                        newTotalTime = maxAvailable;
                        newWidthPercentage = (newTotalTime * 100) / this.options.maxTime;
                        newWidth = (containerWidth * newWidthPercentage) / 100;
                    }
                    
                    this.setStyle('width', `${newWidthPercentage}%`);
                    this.currentTotalTime = newTotalTime;
                    this.totalTimeDiv.write(newTotalTime.toString());
                    
                    this.emit('update', {
                        categoryId: this.category.category_id,
                        newTime: newTotalTime
                    });
                };
                
                const onMouseUp = () => {
                    
                    this.documentBlock.element.removeEventListener('mousemove', onMouseMove);
                    this.documentBlock.element.removeEventListener('mouseup', onMouseUp);
                };
                
                this.documentBlock.element.addEventListener('mousemove', onMouseMove);
                this.documentBlock.element.addEventListener('mouseup', onMouseUp);
            });
        }
    }
    
    /*
    **
    **
    */
    public updateTotalTime(newTime: number) : void {
        
        this.currentTotalTime = newTime;
        this.totalTimeDiv.write(newTime.toString());
        const newWidthPercentage = (newTime * 100) / this.options.maxTime;
        this.setStyle('width', `${newWidthPercentage}%`);
    }
}