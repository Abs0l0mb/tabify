'use strict';

import {
    Div,
    ClientLocation,
    Tools
} from '@src/classes';

import VanillaCalendar from 'vanilla-calendar-pro';

export interface DateSelectionOption {
    date?: Date
}

export class DateSelection extends Div {

    private closeZone: Div;
    private box: Div;
    private vanillaCalendar: VanillaCalendar;
    private date: Date | null;

    constructor(private x: number, private y: number, private options: DateSelectionOption = {}) {
        
        super('date-selection', ClientLocation.get().block);

        this.closeZone = new Div('close-zone', this).onNative('click', this.hide.bind(this));

        this.box = new Div('box', this).setStyles({
            left: '0px',
            top: '0px'
        });

        const vanillaCalendarOptions : any = {
            actions: {
                clickDay: this.onDayClick.bind(this)
            }
        }

        if (this.options.date) {

            vanillaCalendarOptions.date = {
                today: this.options.date
            }
        }

        this.vanillaCalendar = new VanillaCalendar(this.box.element, vanillaCalendarOptions);

        this.vanillaCalendar.init();

        this.show();
    }

    /*
    **
    **
    */
    private onDayClick(element: any, data: any) : void {

        if (!data.selectedDates[0]) {
            this.hide();
            return;
        }

        this.date = new Date(data.selectedDates[0]);
        this.date.setHours(8);
        
        this.emit('date', this.date);

        this.hide();
    }

    /*
    **
    **
    */
    private async show() : Promise<void> {

        await Tools.sleep(50);
        
        let x = this.x;
        let y = this.y;

        if (x + this.box.element.offsetWidth >= document.body.scrollWidth) {
            x -= this.box.element.offsetWidth;
            if (x < 0)
                x = 20;
        }

        if (y + this.box.element.offsetHeight >= document.body.scrollHeight) {
            y -= this.box.element.offsetHeight;
            if (y < 0)
                y = 20;
        }
        
        this.box.setStyles({
            left: `${x}px`,
            top: `${y}px`
        });

        this.setData('displayed', 1);
    }

    /*
    **
    **
    */
    public async hide() : Promise<void> {

        this.emit('hide');
        
        this.setData('displayed', 0);

        await Tools.sleep(250);
        
        this.delete();
    }
}