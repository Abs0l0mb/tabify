'use strict';

import {
    Block,
    Div,
    Tools
} from '@src/classes';

export interface TabData {
    text: string;
    event: string;
}

export class TabsView extends Div {

    private tabsContainer: Div;
    private tabs: Div[] = [];
    public view: Div;
    
    constructor(public tabsData: TabData[], parent: Block) {

        super('tabs-view', parent);

        this.tabsContainer = new Div('tabs', this);
        this.view = new Div('view', this);

        for (let tabData of tabsData) {
            
            let tab = new Div('tab', this.tabsContainer)
                .write(tabData.text)
                .onNative('click', () => {
                    this.select(tab.uid);
                });
            
            tab.publicData = tabData;

            this.tabs.push(tab);
        }

        (async function() {
            await Tools.sleep(10);
            this.select(this.tabs[0].uid);
        }.bind(this)())
    }

    /*
    **
    **
    */
    private select(uid: string) : void {

        for (let tab of this.tabs) {

            if (tab.uid === uid) {

                tab.setData('selected', 1);

                this.emit(tab.publicData.event);
            }
            else
                tab.setData('selected', 0);
        }
    }
}