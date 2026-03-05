'use strict';

import {
    Div,
    ClientLocation,
    Tools,
    Router
} from '@src/classes';

export class Page extends Div {

    protected router: Router = ClientLocation.get().router;

    constructor(title?: string, extraClassName?: string) {

        if (title)
            Page.setTitle(title);
        
        let classNames = ['page'];
        if (extraClassName)
            classNames.push(extraClassName);
        
        super(classNames.join(' '));
        
        setTimeout(function() {
            this.setData('state', 1);
        }.bind(this), 25);

        ClientLocation.get().ready();
    }

    /*
    **
    **
    */
    static setTitle(title: string) : void {

        let titleTag = document.getElementsByTagName('title')[0];

        if (!titleTag)
            return;
        
        title = title ? `${ClientLocation.get().title} - ${title}` : ClientLocation.get().title;

        titleTag.innerHTML = title;
    }

    /*
    **
    **
    */
    public async onLeave() : Promise<void> {

        return new Promise(async function(resolve) {
            this.setData('state', 2);
            await Tools.sleep(180);
            resolve();
        }.bind(this));
    }
}