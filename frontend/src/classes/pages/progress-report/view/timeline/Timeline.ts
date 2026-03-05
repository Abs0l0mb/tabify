import { 
    Div,
    TimelineCategoryCard
} from '@src/classes';

export interface TimelineOptions {
    editable: boolean;
    maxTime: number
}

export class Timeline extends Div {

    public timelineCards: Record<number, TimelineCategoryCard> = {};

    constructor(data: Record<number, any>, parent: Div, private options: TimelineOptions) {

        super('timeline', parent);

        for (const key in data) {

            setTimeout(() => {

                const categoryData = data[key];

                this.timelineCards[key] = new TimelineCategoryCard(categoryData, options, this);
                
                this.timelineCards[key].on('update', (data) => {
                    this.emit('update', data);
                });

            }, parseInt(key, 10) * 100); 
        }
    }

    /*
    **
    **
    */
    public updateCategoryTime(categoryId: number, newTime: number) : void {

        if (this.timelineCards[categoryId])
            this.timelineCards[categoryId].updateTotalTime(newTime);
    }
}


