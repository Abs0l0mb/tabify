'use strict';

import {
    Page,
    Div,
    Tools
} from '@src/classes';

export class TitledPage extends Page {

    public head: Div;
    private titleContainer: Div;
    public actions: Div;
    public content: Div;

    constructor(title?: string, extraClassName?: string) {

        super(title, extraClassName);

        this.addClass('titled');

        this.head = new Div('page-head', this);

        this.titleContainer = new Div('title', this.head);
        
        if (title)
            this.setTitle(title);
            
        this.actions = new Div('actions', this.head);
        this.content = new Div('content', this);
    }

    /*
    **
    **
    */
    protected async setTitle(title: string) : Promise<void> {

        Page.setTitle(title);

        this.titleContainer.setData('displayed', 0);
        this.titleContainer.write(title);

        await Tools.sleep(50);

        this.titleContainer.setData('displayed', 1);
    }

    /*
    **
    **
    */
    public addTitleClass(className: string) {

        this.titleContainer.addClass(className);
    }
}